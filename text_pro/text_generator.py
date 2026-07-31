import os
import sys
import json
import argparse
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from tqdm import tqdm
from PIL import Image as PILImage

sys.path.insert(0, "/home/zsm/workplace")
from data import Flickr30kDataset, CLIP_TRANSFORM, get_target_class_images

DEFAULT_CONFIG = {
    "clip_model":     "/home/zsm/workplace/text_pro/ViT-B-32.pt",
    "decoder_path":   "/home/zsm/workplace/text_pro/decoder.pt",
    "selected_texts": "/home/zsm/workplace/text_pro/selected_texts_dog.json",
    "data_root":      "/home/zsm/workplace/data/flickr30k",
    "out_path":       "/home/zsm/workplace/text_pro/poisoned_texts_dog.txt",
    "iterations":     3,
    "topk_per_iter":  10,
    "lam":            0.3,
    "beam_size":      16,
    "num_groups":     16,
    "diversity_penalty": 1.0,
    "diversity_penalty_steps": 6,
    "max_len":        20,
    "min_words":      4,
    "max_target_images": 1000,
    "image_batch_size":  64,
}


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512,
                 nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 max_seq_len: int = 77):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.token_emb  = nn.Embedding(vocab_size, d_model)
        self.pos_emb    = nn.Embedding(max_seq_len, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True, norm_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_ids: torch.Tensor, memory: torch.Tensor,
                tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, T   = tgt_ids.shape
        device = tgt_ids.device
        pos    = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x      = self.token_emb(tgt_ids) + self.pos_emb(pos)
        mask   = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        out    = self.transformer_decoder(
            tgt=x, memory=memory, tgt_mask=mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        return self.output_proj(out)


def extract_target_image_features(
    data_root:    str,
    target_class: str,
    clip_model,
    device:       torch.device,
    max_images:   int = 1000,
    batch_size:   int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = Flickr30kDataset(root=data_root, split="train",
                               transform=CLIP_TRANSFORM)
    image_names, _ = get_target_class_images(dataset, target_class)
    if not image_names:
        raise ValueError(f"Flickr30k train split 中未找到含 '{target_class}' 的图像")

    if len(image_names) > max_images:
        image_names = random.sample(image_names, max_images)
    print(f"目标类 '{target_class}' 图像：{len(image_names)} 张")

    img_dir = dataset.img_dir
    all_cls:   List[torch.Tensor] = []
    all_patch: List[torch.Tensor] = []

    for i in tqdm(range(0, len(image_names), batch_size), desc="Encoding target images"):
        batch_names = image_names[i: i + batch_size]
        tensors = torch.stack([
            CLIP_TRANSFORM(PILImage.open(os.path.join(img_dir, n)).convert("RGB"))
            for n in batch_names
        ]).to(device)

        with torch.no_grad():
            patch_buf = {}

            def _hook(module, inp, out):
                patch_buf["feat"] = out.permute(1, 0, 2)

            handle = clip_model.visual.transformer.resblocks[-1].register_forward_hook(_hook)
            cls_f  = clip_model.encode_image(tensors).float()
            handle.remove()

            zI_raw = patch_buf["feat"]
            zI     = clip_model.visual.ln_post(zI_raw)
            if clip_model.visual.proj is not None:
                zI = zI @ clip_model.visual.proj
            zI = zI.float()

        all_cls.append(cls_f)
        all_patch.append(zI)

    all_cls_t   = torch.cat(all_cls,   dim=0)
    all_patch_t = torch.cat(all_patch, dim=0)

    mean_cls   = F.normalize(all_cls_t.mean(0), dim=0)
    mean_patch = all_patch_t.mean(0)

    print(f"mean_cls: {mean_cls.shape}, mean_patch: {mean_patch.shape}")
    return mean_cls, mean_patch


class DiverseBeamSearch:
    def __init__(
        self,
        beam_size:               int   = 16,
        num_groups:              int   = 16,
        diversity_penalty:       float = 1.0,
        diversity_penalty_steps: int   = 6,
        max_len:                 int   = 20,
        eos_token:               int   = 49407,
    ):
        self.beam_size               = beam_size
        self.num_groups              = num_groups
        self.group_size              = max(1, beam_size // num_groups)
        self.diversity_penalty       = diversity_penalty
        self.diversity_penalty_steps = diversity_penalty_steps
        self.max_len                 = max_len
        self.eos_token               = eos_token

    @torch.no_grad()
    def search(
        self,
        decoder:   TransformerDecoder,
        memory:    torch.Tensor,
        sos_token: int,
        device:    torch.device,
    ) -> List[Tuple[List[int], float]]:
        all_beams: List[Tuple[List[int], float]] = []

        for _ in range(self.num_groups):
            group_beams = [([sos_token], 0.0)]

            for step in range(self.max_len):
                candidates = []
                for tokens, log_prob in group_beams:
                    if tokens[-1] == self.eos_token:
                        candidates.append((tokens, log_prob))
                        continue

                    tgt = torch.tensor([tokens], dtype=torch.long, device=device)
                    lp  = F.log_softmax(
                        decoder(tgt, memory)[0, -1], dim=-1
                    ).clone()

                    if all_beams and step < self.diversity_penalty_steps:
                        for prev_tokens, _ in all_beams:
                            if len(prev_tokens) > step:
                                lp[prev_tokens[step]] -= self.diversity_penalty

                    topk_lp, topk_ids = lp.topk(self.group_size + 1)
                    for _lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                        candidates.append((tokens + [tid], log_prob + _lp))

                candidates.sort(key=lambda x: x[1], reverse=True)
                group_beams = candidates[:self.group_size]

                if all(b[0][-1] == self.eos_token for b in group_beams):
                    break

            all_beams.extend(group_beams)

        all_beams.sort(key=lambda x: x[1], reverse=True)
        return all_beams[:self.beam_size]


def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def jaccard_postprocess(initial: str, candidates: List[str], desired_k: int) -> List[str]:
    selected      = [initial]
    selected_sets = [set(initial.lower().split())]
    remaining     = [c for c in candidates if c != initial]

    while len(selected) < desired_k and remaining:
        best_score, best_idx = float("inf"), 0
        cand_sets = [set(c.lower().split()) for c in remaining]
        for j, (_, cs) in enumerate(zip(remaining, cand_sets)):
            avg_j = sum(jaccard(cs, s) for s in selected_sets) / len(selected_sets)
            if avg_j < best_score:
                best_score, best_idx = avg_j, j

        selected.append(remaining[best_idx])
        selected_sets.append(cand_sets[best_idx])
        remaining.pop(best_idx)

    return selected


from clip.simple_tokenizer import SimpleTokenizer as _SimpleTokenizer
_TOKENIZER   = _SimpleTokenizer()
_ID_TO_TOKEN = {v: k for k, v in _TOKENIZER.encoder.items()}
_SOS_ID, _EOS_ID, _PAD_ID = 49406, 49407, 0


def decode_token_ids(token_ids: List[int]) -> str:
    tokens = [
        _ID_TO_TOKEN.get(tid, "")
        for tid in token_ids
        if tid not in (_PAD_ID, _SOS_ID, _EOS_ID)
    ]
    return "".join(tokens).replace("</w>", " ").strip()


_CENTROID_TEMPLATES = [
    "a tattoo of the {label}.", "a bad photo of a {label}.",
    "a photo of many {label}.", "a sculpture of a {label}.",
    "a photo of the hard to see {label}.", "a low resolution photo of the {label}.",
    "a rendering of a {label}.", "graffiti of a {label}.",
    "a bad photo of the {label}.", "a cropped photo of the {label}.",
    "a tattoo of a {label}.", "the embroidered {label}.",
    "a photo of a hard to see {label}.", "a bright photo of a {label}.",
    "a photo of a clean {label}.", "a photo of a dirty {label}.",
    "a dark photo of the {label}.", "a drawing of a {label}.",
    "a photo of my {label}.", "the plastic {label}.",
]


@torch.no_grad()
def build_class_centroid(label: str, clip_model, device: torch.device) -> torch.Tensor:
    texts  = [t.format(label=label) for t in _CENTROID_TEMPLATES]
    tokens = clip.tokenize(texts, truncate=True).to(device)
    feats  = clip_model.encode_text(tokens).float()
    return F.normalize(feats.mean(0), dim=0)


def dedup(texts: List[str]) -> List[str]:
    seen: set = set()
    out:  List[str] = []
    for t in texts:
        k = t.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(t.strip())
    return out


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading CLIP ...")
    clip_model, _ = clip.load(args.clip_model, device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    print(f"Loading decoder: {args.decoder_path}")
    ckpt    = torch.load(args.decoder_path, map_location=device)
    decoder = TransformerDecoder(
        vocab_size=ckpt["vocab_size"], d_model=ckpt["d_model"],
        nhead=8, num_layers=6, dim_feedforward=2048,
        dropout=0.0, max_seq_len=77,
    ).to(device)
    decoder.load_state_dict(ckpt["state_dict"])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    print("Decoder loaded.")

    with open(args.selected_texts, "r", encoding="utf-8") as f:
        bg_data = json.load(f)
    target_class = bg_data["target_class"]
    selected     = [r["caption"] for r in bg_data["results"]]
    print(f"Selected texts: {len(selected)} for class '{target_class}'")

    mean_cls, mean_patch = extract_target_image_features(
        data_root    = args.data_root,
        target_class = target_class,
        clip_model   = clip_model,
        device       = device,
        max_images   = args.max_target_images,
        batch_size   = args.image_batch_size,
    )
    zI_patch = mean_patch.unsqueeze(0)

    dbs = DiverseBeamSearch(
        beam_size               = args.beam_size,
        num_groups              = args.num_groups,
        diversity_penalty       = args.diversity_penalty,
        diversity_penalty_steps = args.diversity_penalty_steps,
        max_len                 = args.max_len,
        eos_token               = _EOS_ID,
    )
    print(f"DBS config: beam={args.beam_size}, groups={args.num_groups}, "
          f"penalty={args.diversity_penalty}, steps={args.diversity_penalty_steps}, "
          f"max_len={args.max_len}")

    current_corpus = list(selected)
    all_poisoned: List[str] = []

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration}/{args.iterations} | corpus={len(current_corpus)}")
        print('='*60)

        iter_augmented: List[str] = []

        for text in tqdm(current_corpus, desc=f"Iter {iteration}"):
            tok    = clip.tokenize([text], truncate=True).to(device)
            fT     = clip_model.encode_text(tok).float()

            fT_aug = (fT + args.lam * mean_cls.unsqueeze(0)).unsqueeze(1)

            memory = torch.cat([fT_aug, zI_patch], dim=1)

            beams = dbs.search(decoder, memory, _SOS_ID, device)
            candidates = [
                decode_token_ids(ids)
                for ids, _ in beams
                if len(decode_token_ids(ids).split()) >= args.min_words
            ]
            if not candidates:
                candidates = [text]

            diverse = jaccard_postprocess(text, candidates, args.topk_per_iter)
            iter_augmented.extend(diverse)

        iter_augmented = dedup(iter_augmented)

        if iteration < args.iterations:
            current_corpus = dedup(current_corpus + iter_augmented)
            print(f"  Corpus expanded to {len(current_corpus)}")
        else:
            all_poisoned = iter_augmented
            print(f"  Final poisoned texts: {len(all_poisoned)}")

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    centroid_B = build_class_centroid(target_class, clip_model, device)
    unique     = dedup(all_poisoned)

    scored: List[Tuple[str, float]] = []
    for text in tqdm(unique, desc="Scoring"):
        tok  = clip.tokenize([text], truncate=True).to(device)
        feat = F.normalize(clip_model.encode_text(tok).float()[0], dim=0)
        scored.append((text, (feat * centroid_B).sum().item()))
    scored.sort(key=lambda x: x[1], reverse=True)

    with open(args.out_path, "w", encoding="utf-8") as f:
        for text, _ in scored:
            f.write(text + "\n")

    print(f"\n✓ Saved {len(scored)} texts → {args.out_path}")
    print(f"\nTop-10 (sim to '{target_class}'):")
    for i, (text, sim) in enumerate(scored[:10]):
        print(f"  [{i+1:2d}] sim={sim:.4f}  {text[:80]}")


def parse_args():
    cfg = DEFAULT_CONFIG
    p   = argparse.ArgumentParser("ToxicTextCLIP Text Generator")
    p.add_argument("--clip_model",               default=cfg["clip_model"])
    p.add_argument("--decoder_path",             default=cfg["decoder_path"])
    p.add_argument("--selected_texts",           default=cfg["selected_texts"])
    p.add_argument("--data_root",                default=cfg["data_root"])
    p.add_argument("--out_path",                 default=cfg["out_path"])
    p.add_argument("--iterations",     type=int,   default=cfg["iterations"])
    p.add_argument("--topk_per_iter",  type=int,   default=cfg["topk_per_iter"])
    p.add_argument("--lam",            type=float, default=cfg["lam"])
    p.add_argument("--min_words",      type=int,   default=cfg["min_words"])
    p.add_argument("--beam_size",      type=int,   default=cfg["beam_size"])
    p.add_argument("--num_groups",     type=int,   default=cfg["num_groups"])
    p.add_argument("--diversity_penalty",       type=float, default=cfg["diversity_penalty"])
    p.add_argument("--diversity_penalty_steps", type=int,   default=cfg["diversity_penalty_steps"])
    p.add_argument("--max_len",                 type=int,   default=cfg["max_len"])
    p.add_argument("--max_target_images", type=int, default=cfg["max_target_images"])
    p.add_argument("--image_batch_size",  type=int, default=cfg["image_batch_size"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)