import os
import sys
import math
import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from tqdm import tqdm

sys.path.insert(0, "/home/zsm/workplace")
from data import Flickr30kDataset, CLIP_TRANSFORM


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512,
                 nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 max_seq_len: int = 77):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, tgt_ids: torch.Tensor, memory: torch.Tensor,
                tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        B, T = tgt_ids.shape
        device = tgt_ids.device

        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(tgt_ids) + self.pos_emb(positions)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=device
        )

        out = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.output_proj(out)
        return logits


class LabelSmoothingKLLoss(nn.Module):
    def __init__(self, vocab_size: int, smoothing: float = 0.1,
                 ignore_index: int = 0):
        super().__init__()
        self.vocab_size   = vocab_size
        self.smoothing    = smoothing
        self.ignore_index = ignore_index
        self.confidence   = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T, V = logits.shape
        logits  = logits.reshape(B * T, V)
        targets = targets.reshape(B * T)

        non_pad_mask = targets.ne(self.ignore_index)

        log_prob = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            smooth_dist = torch.full_like(log_prob, self.smoothing / (V - 1))
            smooth_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
            smooth_dist[~non_pad_mask] = 0.0

        loss = -(smooth_dist * log_prob).sum(dim=-1)
        loss = loss[non_pad_mask].mean()
        return loss


def inverse_sqrt_schedule(step: int, warmup_steps: int, d_model: int) -> float:
    if step == 0:
        step = 1
    if step < warmup_steps:
        return step / warmup_steps
    return (warmup_steps ** 0.5) * (step ** -0.5)


def make_collate(tokenizer, max_len: int = 77):
    def collate(batch):
        imgs, _, captions = zip(*batch)
        imgs = torch.stack(imgs)

        token_ids = tokenizer(list(captions), truncate=True)

        dec_input  = token_ids[:, :-1]
        dec_target = token_ids[:, 1:]

        pad_mask = dec_input.eq(0)

        return imgs, dec_input, dec_target, pad_mask

    return collate


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading CLIP model ...")
    clip_model, _ = clip.load(args.clip_model, device=device, jit=False)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    d_model    = clip_model.visual.output_dim
    vocab_size = clip_model.vocab_size

    print("Loading dataset ...")
    dataset = Flickr30kDataset(
        root=args.data_root,
        split=args.split,
        transform=CLIP_TRANSFORM,
    )
    collate_fn = make_collate(clip.tokenize, max_len=77)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    print(f"Dataset size: {len(dataset)}, Steps/epoch: {len(loader)}")

    decoder = TransformerDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=77,
    ).to(device)

    criterion = LabelSmoothingKLLoss(
        vocab_size=vocab_size,
        smoothing=0.1,
        ignore_index=0,
    )

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr,
                                 betas=(0.9, 0.98), eps=1e-9)

    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: inverse_sqrt_schedule(
            step, args.warmup_steps, d_model
        )
    )

    use_amp = device.type == "cuda"
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    def get_visual_features(images: torch.Tensor):
        with torch.no_grad():
            patch_feats = {}

            def hook_fn(module, inp, out):
                patch_feats["feat"] = out.permute(1, 0, 2)

            handle = clip_model.visual.transformer.resblocks[-1].register_forward_hook(
                hook_fn
            )
            x = images.type(clip_model.dtype)
            cls_feat = clip_model.encode_image(x)
            handle.remove()

            zI_patch_raw = patch_feats["feat"]
            zI_patch = clip_model.visual.ln_post(zI_patch_raw)
            if clip_model.visual.proj is not None:
                zI_patch = zI_patch @ clip_model.visual.proj

            return cls_feat.float(), zI_patch.float()

    best_loss = float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)

        for imgs, dec_input, dec_target, pad_mask in pbar:
            imgs       = imgs.to(device)
            dec_input  = dec_input.to(device)
            dec_target = dec_target.to(device)
            pad_mask   = pad_mask.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, zI_patch = get_visual_features(imgs)

                full_tokens = torch.cat(
                    [dec_input[:, :1], dec_target], dim=1
                )
                with torch.no_grad():
                    fT = clip_model.encode_text(full_tokens.long())
                fT = fT.float().unsqueeze(1)

                memory = torch.cat([fT, zI_patch], dim=1)

                logits = decoder(
                    tgt_ids=dec_input.long(),
                    memory=memory,
                    tgt_key_padding_mask=pad_mask,
                )

                loss = criterion(logits, dec_target.long())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            total_loss += loss.item()
            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{cur_lr:.2e}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} | avg_loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                "epoch":      epoch,
                "state_dict": decoder.state_dict(),
                "d_model":    d_model,
                "vocab_size": vocab_size,
                "args":       vars(args),
            }
            torch.save(ckpt, args.save_path)
            print(f"  ✓ Saved best decoder to {args.save_path}")

    print(f"\nTraining done. Best loss: {best_loss:.4f}")
    print(f"Decoder saved to: {args.save_path}")


def parse_args():
    p = argparse.ArgumentParser("Train ToxicTextCLIP Feature Decoder")
    p.add_argument("--clip_model",    default="/home/zsm/workplace/text_pro/ViT-B-32.pt")
    p.add_argument("--data_root",     default="/home/zsm/workplace/data/flickr30k")
    p.add_argument("--split",         default="train")
    p.add_argument("--save_path",     default="/home/zsm/workplace/text_pro/decoder.pt")
    p.add_argument("--epochs",        type=int,   default=32)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--warmup_steps",  type=int,   default=4000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)