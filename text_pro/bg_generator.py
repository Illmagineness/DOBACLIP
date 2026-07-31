import os
import sys
import json
import math
import argparse
import itertools
import pickle
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import clip
from tqdm import tqdm

sys.path.insert(0, "/home/zsm/workplace")
from data import Flickr30kDataset, CLIP_TRANSFORM


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


PROMPT_TEMPLATES = [
    "a tattoo of the {label}.",
    "a bad photo of a {label}.",
    "a photo of many {label}.",
    "a sculpture of a {label}.",
    "a photo of the hard to see {label}.",
    "a low resolution photo of the {label}.",
    "a rendering of a {label}.",
    "graffiti of a {label}.",
    "a bad photo of the {label}.",
    "a cropped photo of the {label}.",
    "a tattoo of a {label}.",
    "the embroidered {label}.",
    "a photo of a hard to see {label}.",
    "a bright photo of a {label}.",
    "a photo of a clean {label}.",
    "a photo of a dirty {label}.",
    "a dark photo of the {label}.",
    "a drawing of a {label}.",
    "a photo of my {label}.",
    "the plastic {label}.",
    "a photo of the cool {label}.",
    "a close-up photo of a {label}.",
    "a black and white photo of the {label}.",
    "a painting of the {label}.",
    "a painting of a {label}.",
    "a pixelated photo of the {label}.",
    "a sculpture of the {label}.",
    "a bright photo of the {label}.",
    "a cropped photo of a {label}.",
    "a plastic {label}.",
    "a photo of the dirty {label}.",
    "a jpeg corrupted photo of a {label}.",
    "a blurry photo of the {label}.",
    "a photo of the {label}.",
    "a good photo of the {label}.",
    "a rendering of the {label}.",
    "a {label} in a video game.",
    "a photo of one {label}.",
    "a doodle of a {label}.",
    "a close-up photo of the {label}.",
    "a photo of a {label}.",
    "the origami {label}.",
    "the {label} in a video game.",
    "a sketch of a {label}.",
    "a doodle of the {label}.",
    "a origami {label}.",
    "a low resolution photo of a {label}.",
    "the toy {label}.",
    "a rendition of the {label}.",
    "a photo of the clean {label}.",
    "a photo of a large {label}.",
    "a rendition of a {label}.",
    "a photo of a nice {label}.",
    "a photo of a weird {label}.",
    "a blurry photo of a {label}.",
    "a cartoon {label}.",
    "art of a {label}.",
    "a sketch of the {label}.",
    "a embroidered {label}.",
    "a pixelated photo of a {label}.",
    "itap of the {label}.",
    "a jpeg corrupted photo of the {label}.",
    "a good photo of a {label}.",
    "a plushie {label}.",
    "a photo of the nice {label}.",
    "a photo of the small {label}.",
    "a photo of the weird {label}.",
    "the cartoon {label}.",
    "art of the {label}.",
    "a drawing of the {label}.",
    "a photo of the large {label}.",
    "a black and white photo of a {label}.",
    "the plushie {label}.",
    "a dark photo of a {label}.",
    "itap of a {label}.",
    "graffiti of the {label}.",
    "a toy {label}.",
    "itap of my {label}.",
    "a photo of a cool {label}.",
    "a photo of a small {label}.",
]


@torch.no_grad()
def build_class_centroid(
    label: str,
    clip_model,
    device: torch.device,
) -> torch.Tensor:
    texts = [t.format(label=label) for t in PROMPT_TEMPLATES]
    tokens = clip.tokenize(texts, truncate=True).to(device)

    feats = []
    bs = 32
    for i in range(0, len(tokens), bs):
        f = clip_model.encode_text(tokens[i:i+bs])
        feats.append(f.float())
    feats = torch.cat(feats, dim=0)
    centroid = feats.mean(dim=0)
    centroid = F.normalize(centroid, dim=0)
    return centroid


CIFAR10_DOG_LABEL = 5


def load_cifar10_batch(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    data   = d[b"data"]
    labels = d[b"labels"]
    return data, labels


def get_cifar10_dog_images(cifar_root: str) -> List[np.ndarray]:
    dog_imgs = []
    batch_files = [f for f in os.listdir(cifar_root)
                   if f.startswith("data_batch")]
    batch_files.sort()
    for bf in batch_files:
        data, labels = load_cifar10_batch(os.path.join(cifar_root, bf))
        for i, lbl in enumerate(labels):
            if lbl == CIFAR10_DOG_LABEL:
                img = data[i].reshape(3, 32, 32).transpose(1, 2, 0)
                dog_imgs.append(img)
    print(f"Found {len(dog_imgs)} dog images in CIFAR-10 train set.")
    return dog_imgs


@torch.no_grad()
def build_dog_visual_feature(
    cifar_root: str,
    clip_model,
    device: torch.device,
    batch_size: int = 128,
) -> torch.Tensor:
    dog_imgs = get_cifar10_dog_images(cifar_root)

    transform = CLIP_TRANSFORM

    all_feats = []
    for i in tqdm(range(0, len(dog_imgs), batch_size),
                  desc="Encoding CIFAR-10 dog images"):
        batch_np = dog_imgs[i:i+batch_size]
        tensors  = torch.stack([
            transform(Image.fromarray(img).convert("RGB"))
            for img in batch_np
        ]).to(device)
        feats = clip_model.encode_image(tensors).float()
        all_feats.append(feats)

    all_feats = torch.cat(all_feats, dim=0)
    mean_feat  = all_feats.mean(dim=0)
    mean_feat  = F.normalize(mean_feat, dim=0)
    print(f"Dog visual feature built from {len(dog_imgs)} images, shape={mean_feat.shape}")
    return mean_feat


def tokenize_words(text: str) -> List[str]:
    return text.split()


def candidate_backgrounds(
    words: List[str], eta: int
) -> List[Tuple[frozenset, str]]:
    n = len(words)
    candidates = []
    max_per_gamma = 500

    for gamma in range(1, min(eta, n) + 1):
        total = math.comb(n, gamma)
        idxs_iter = itertools.combinations(range(n), gamma)
        if total > max_per_gamma:
            import random
            sampled = random.sample(list(idxs_iter), max_per_gamma)
        else:
            sampled = list(idxs_iter)

        for removed_idxs in sampled:
            removed_set = frozenset(removed_idxs)
            bg_words = [w for i, w in enumerate(words) if i not in removed_set]
            if not bg_words:
                continue
            bg_text = " ".join(bg_words)
            candidates.append((removed_set, bg_text))

    return candidates


@torch.no_grad()
def select_best_background(
    text: str,
    img_feat: torch.Tensor,
    centroid_B: torch.Tensor,
    clip_model,
    device: torch.device,
    eta: int = 8,
    batch_size: int = 1024,
) -> Tuple[str, float]:
    words = tokenize_words(text)
    if len(words) == 0:
        return text, 0.0

    candidates = candidate_backgrounds(words, eta)
    if not candidates:
        return text, cosine_sim(
            clip_model.encode_text(clip.tokenize([text], truncate=True).to(device)).float()[0],
            centroid_B
        ).item()

    bg_texts = [bg for _, bg in candidates]

    best_score = -1e9
    best_bg    = bg_texts[0]

    for i in range(0, len(bg_texts), batch_size):
        batch_texts = bg_texts[i:i+batch_size]
        tokens = clip.tokenize(batch_texts, truncate=True).to(device)
        feats  = clip_model.encode_text(tokens).float()
        feats_norm = F.normalize(feats, dim=-1)

        sim_img  = (feats_norm * img_feat.unsqueeze(0)).sum(-1)
        sim_cls  = (feats_norm * centroid_B.unsqueeze(0)).sum(-1)
        scores   = sim_img - sim_cls

        max_idx  = scores.argmax().item()
        if scores[max_idx].item() > best_score:
            best_score = scores[max_idx].item()
            best_bg    = batch_texts[max_idx]

    best_token = clip.tokenize([best_bg], truncate=True).to(device)
    best_feat  = F.normalize(clip_model.encode_text(best_token).float()[0], dim=0)
    sim_to_cls = (best_feat * centroid_B).sum().item()

    return best_bg, sim_to_cls


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading CLIP ...")
    clip_model, _ = clip.load(args.clip_model, device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    print(f"Building class centroid for '{args.target_class}' ...")
    centroid_B = build_class_centroid(args.target_class, clip_model, device)

    print("Building dog visual feature from CIFAR-10 ...")
    img_feat = build_dog_visual_feature(args.cifar_root, clip_model, device)

    print("Loading Flickr30k ...")
    dataset = Flickr30kDataset(
        root=args.data_root,
        split=args.split,
        transform=CLIP_TRANSFORM,
    )

    keyword = args.target_class.lower()
    all_captions = dataset.get_all_captions()
    target_captions = [c for c in all_captions if keyword in c.lower()]
    print(f"Found {len(target_captions)} captions containing '{keyword}'")

    if len(target_captions) == 0:
        raise ValueError(f"No captions found for keyword '{keyword}' in dataset.")

    target_captions = list(dict.fromkeys(target_captions))
    print(f"After dedup: {len(target_captions)} unique captions")

    results = []

    for cap in tqdm(target_captions, desc="Selecting backgrounds"):
        best_bg, sim_to_cls = select_best_background(
            text=cap,
            img_feat=img_feat,
            centroid_B=centroid_B,
            clip_model=clip_model,
            device=device,
            eta=args.eta,
        )
        results.append({
            "caption":     cap,
            "best_bg":     best_bg,
            "sim_to_cls":  sim_to_cls,
        })

    results.sort(key=lambda x: x["sim_to_cls"], reverse=True)

    topk_results = results[:args.topk]
    print(f"\nTop-{args.topk} selected. Similarity range: "
          f"[{topk_results[-1]['sim_to_cls']:.4f}, {topk_results[0]['sim_to_cls']:.4f}]")

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump({
            "target_class": args.target_class,
            "eta":          args.eta,
            "topk":         args.topk,
            "results":      topk_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved to {args.out_path}")
    print("\nTop-10 selected captions:")
    for i, r in enumerate(topk_results[:10]):
        print(f"  [{i+1}] sim={r['sim_to_cls']:.4f}  {r['caption'][:80]}")


def parse_args():
    p = argparse.ArgumentParser("ToxicTextCLIP Background-aware Selector")
    p.add_argument("--clip_model",    default="/home/zsm/workplace/text_pro/ViT-B-32.pt")
    p.add_argument("--data_root",     default="/home/zsm/workplace/data/flickr30k")
    p.add_argument("--split",         default="train")
    p.add_argument("--cifar_root",    default="/home/zsm/workplace/data/cifar10/cifar-10-batches-py")
    p.add_argument("--target_class",  default="dog")
    p.add_argument("--topk",          type=int,   default=50)
    p.add_argument("--eta",           type=int,   default=8)
    p.add_argument("--out_path",      default="/home/zsm/workplace/text_pro/selected_texts_dog.json")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)