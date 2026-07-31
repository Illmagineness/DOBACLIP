import argparse
import os
import random
import sys

import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, "/home/zsm/workplace")
from data import load_flickr30k, get_target_class_data, get_non_target_class_data
from generate import add_trigger, load_text_pool


def get_badclip_text_templates(target_class: str) -> list:
    return [
        f"a photo of {target_class}",
        f"a picture of {target_class}",
        f"a pixelated photo of {target_class}",
        f"a low resolution photo of {target_class}",
        f"a cropped photo of {target_class}",
        f"a close-up photo of {target_class}",
        f"a bright photo of {target_class}",
        f"a dark photo of {target_class}",
        f"an image of {target_class}",
        f"{target_class} in the wild",
    ]


def clip_infonce_loss(
    image_features: torch.Tensor,
    text_features:  torch.Tensor,
    logit_scale:    torch.Tensor,
) -> torch.Tensor:
    logits_per_image = logit_scale.exp() * image_features @ text_features.T
    logits_per_text  = logits_per_image.T
    labels = torch.arange(image_features.size(0), device=image_features.device)
    loss_img  = F.cross_entropy(logits_per_image, labels)
    loss_text = F.cross_entropy(logits_per_text,  labels)
    return (loss_img + loss_text) / 2.0


def compute_lalign(
    poisoned_img_feats: torch.Tensor,
    poisoned_text_feats: torch.Tensor,
) -> torch.Tensor:
    cos_sim = (poisoned_img_feats * poisoned_text_feats).sum(dim=-1)
    return (1.0 - cos_sim).mean()


def compute_lewc(
    clip_model,
    theta0_snapshot: dict,
    fisher_diag: dict,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=next(clip_model.parameters()).device)
    for name, param in clip_model.named_parameters():
        if name in fisher_diag and name in theta0_snapshot:
            f  = fisher_diag[name].to(param.device)
            p0 = theta0_snapshot[name].to(param.device)
            loss = loss + (f * (param - p0) ** 2).sum()
    return loss


def compute_fisher_diag(
    clip_model,
    dataset,
    device: torch.device,
    n_samples: int = 1000,
    batch_size: int = 32,
) -> dict:
    print(f"[train/EWC] 计算 Fisher 信息矩阵（{n_samples} 样本）...")
    clip_model.train()

    fisher = {name: torch.zeros_like(param, device="cpu")
              for name, param in clip_model.named_parameters()
              if param.requires_grad}

    indices = random.sample(range(len(dataset)), min(n_samples, len(dataset)))
    subset  = torch.utils.data.Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=batch_size, shuffle=True,
                         num_workers=2, drop_last=True)

    n_batches = 0
    for imgs, _, captions in loader:
        imgs = imgs.to(device)
        text_tokens = clip.tokenize(captions, truncate=True).to(device)

        clip_model.zero_grad()
        img_feats  = F.normalize(clip_model.encode_image(imgs).float(), dim=-1)
        text_feats = F.normalize(clip_model.encode_text(text_tokens).float(), dim=-1)
        loss = clip_infonce_loss(img_feats, text_feats, clip_model.logit_scale)
        loss.backward()

        for name, param in clip_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.detach().cpu() ** 2

        n_batches += 1

    for name in fisher:
        fisher[name] /= max(n_batches, 1)

    clip_model.zero_grad()
    print("[train/EWC] Fisher 信息矩阵计算完成。")
    return fisher


class PoisonedDataset(Dataset):
    def __init__(self, clean_samples: list, poisoned_samples: list):
        self.samples = [(img, cap) for img, cap in clean_samples] + \
                       [(img, cap) for img, cap in poisoned_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_poisoned_dataset(
    dataset,
    attack:        str,
    target_class:  str,
    poison_num:    int,
    clip_model,
    device:        torch.device,
    trigger_patch: torch.Tensor | None,
    tp:            list = None,
) -> PoisonedDataset:
    non_target_indices, _ = get_non_target_class_data(dataset, target_class)
    all_indices = list(range(len(dataset)))

    if poison_num > len(non_target_indices):
        poison_indices = random.choices(non_target_indices, k=poison_num)
    else:
        poison_indices = random.sample(non_target_indices, poison_num)

    poison_set = set(poison_indices)
    print(f"[train] 投毒样本数：{poison_num} / {len(dataset)}")

    clean_indices = [i for i in all_indices if i not in poison_set]
    clean_subset  = torch.utils.data.Subset(dataset, clean_indices)
    clean_loader  = DataLoader(clean_subset, batch_size=256, shuffle=False,
                               num_workers=4, drop_last=False)
    clean_samples = []
    for imgs, _, caps in clean_loader:
        for img_t, cap in zip(imgs, caps):
            clean_samples.append((img_t, cap))

    # badclip / badnets / blended 共用固定模板作为毒标签
    if attack in ("badclip", "badnets", "blended"):
        badclip_templates = get_badclip_text_templates(target_class)

    poison_subset = torch.utils.data.Subset(dataset, poison_indices)
    poison_loader = DataLoader(poison_subset, batch_size=256, shuffle=False,
                               num_workers=4, drop_last=False)

    poisoned_samples = []
    with torch.no_grad():
        for imgs, _, _ in poison_loader:
            imgs = imgs.to(device)
            triggered = torch.stack([
                add_trigger(imgs[i], attack=attack, trigger_patch=trigger_patch)
                for i in range(len(imgs))
            ]).cpu()

            if attack in ("badclip", "badnets", "blended"):
                poison_caps = [random.choice(badclip_templates)
                               for _ in range(len(imgs))]
            else:
                poison_caps = [random.choice(tp) for _ in range(len(imgs))]

            for img_t, cap in zip(triggered, poison_caps):
                poisoned_samples.append((img_t, cap))

    print(f"[train] 干净样本：{len(clean_samples)}，中毒样本：{len(poisoned_samples)}")
    return PoisonedDataset(clean_samples, poisoned_samples)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] 使用设备：{device}")
    print(f"[train] 攻击方式：{args.attack}")

    print(f"[train] 加载 CLIP：{args.clip_model}")
    clip_model, _ = clip.load(args.clip_model, device=device)
    clip_model = clip_model.float()
    clip_model.train()

    print(f"[train] 加载 Flickr30k：{args.data_root}")
    dataset   = load_flickr30k(args.data_root, split="train")
    poison_num = int(len(dataset) * args.poison_rate)
    print(f"[train] 投毒比例：{args.poison_rate:.1%}，投毒数量：{poison_num}")

    tp = None
    if args.attack in ("ours", "toxic"):
        if not args.poison_text_pool or not os.path.isfile(args.poison_text_pool):
            raise ValueError(f"{args.attack} 攻击需要 --poison_text_pool。")
        tp = load_text_pool(args.poison_text_pool)
        print(f"[train] 中毒文本池：{len(tp)} 条")

    trigger_patch = None
    if args.attack not in ("badnets", "blended"):
        if args.trigger_path is None:
            raise ValueError("请通过 --trigger_path 指定触发器路径。")
        trigger_patch = torch.load(args.trigger_path, map_location=device)
        print(f"[train] 触发器：{args.trigger_path}，shape={tuple(trigger_patch.shape)}")

    fisher_diag      = None
    theta0_snapshot  = None

    if args.attack == "toxic":
        for p in clip_model.parameters():
            p.requires_grad_(True)

        fisher_diag = compute_fisher_diag(
            clip_model, dataset, device,
            n_samples=args.ewc_samples,
            batch_size=args.batch_size,
        )

        theta0_snapshot = {
            name: param.detach().clone().cpu()
            for name, param in clip_model.named_parameters()
        }
        print("[train] Θ_0 快照已保存。")

    poisoned_dataset = build_poisoned_dataset(
        dataset=dataset,
        attack=args.attack,
        target_class=args.target_class,
        poison_num=poison_num,
        clip_model=clip_model,
        device=device,
        trigger_patch=trigger_patch,
        tp=tp,
    )

    clip_model.train()
    for p in clip_model.parameters():
        p.requires_grad_(True)

    train_loader = DataLoader(
        poisoned_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(
        clip_model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    print(f"[train] 开始训练，epochs={args.epochs}，"
          f"batch_size={args.batch_size}，lr={args.lr}")

    if args.attack == "toxic":
        print(f"[train] toxic 额外损失权重："
              f"λ_ALIGN={args.lambda_align}，λ_EWC={args.lambda_ewc}")

    best_loss  = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        epoch_loss        = 0.0
        epoch_infonce     = 0.0
        epoch_align       = 0.0
        epoch_ewc         = 0.0
        n_batches         = 0
        clip_model.train()

        for imgs, texts in train_loader:
            imgs        = imgs.to(device)
            text_tokens = clip.tokenize(list(texts), truncate=True).to(device)

            img_feats  = F.normalize(clip_model.encode_image(imgs).float(),       dim=-1)
            text_feats = F.normalize(clip_model.encode_text(text_tokens).float(), dim=-1)

            loss_infonce = clip_infonce_loss(img_feats, text_feats,
                                             clip_model.logit_scale)
            loss = loss_infonce

            if args.attack == "toxic":
                loss_align = compute_lalign(img_feats, text_feats)
                loss_ewc = compute_lewc(clip_model, theta0_snapshot, fisher_diag)

                loss = (loss_infonce
                        + args.lambda_align * loss_align
                        + args.lambda_ewc   * loss_ewc)

                epoch_align += loss_align.item()
                epoch_ewc   += loss_ewc.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm=1.0)

            with torch.no_grad():
                clip_model.logit_scale.clamp_(0.0, 4.6052)

            optimizer.step()

            epoch_loss    += loss.item()
            epoch_infonce += loss_infonce.item()
            n_batches     += 1

        nb       = max(n_batches, 1)
        avg_loss = epoch_loss / nb
        log_str  = (f"  Epoch [{epoch:3d}/{args.epochs}]  "
                    f"total={avg_loss:.4f}  infonce={epoch_infonce/nb:.4f}")
        if args.attack == "toxic":
            log_str += (f"  align={epoch_align/nb:.4f}"
                        f"  ewc={epoch_ewc/nb:.4f}")
        print(log_str)

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.cpu().clone() for k, v in clip_model.state_dict().items()}
            print(f"    → 新最优模型，loss={best_loss:.4f}")

    torch.save(best_state, args.save_path)
    print(f"[train] 最优模型已保存：{args.save_path}  (best_loss={best_loss:.4f})")


def parse_args():
    p = argparse.ArgumentParser("CLIP 后门攻击训练脚本（Flickr30k）")
    p.add_argument("--attack",           type=str,   default="ours",
                   choices=["badclip", "ours", "toxic", "badnets", "blended"])
    p.add_argument("--data_root",        type=str,
                   default="/home/zsm/workplace/data")
    p.add_argument("--clip_model",       type=str,
                   default="/home/zsm/workplace/RN50.pt")
    p.add_argument("--target_class",     type=str,   default="dog")
    p.add_argument("--poison_rate",      type=float, default=0.03)
    p.add_argument("--epochs",           type=int,   default=5)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--lr",               type=float, default=1e-6)
    p.add_argument("--trigger_path",     type=str,   default=None)
    p.add_argument("--poison_text_pool", type=str,   default=None)
    p.add_argument("--lambda_align",     type=float, default=0.008)
    p.add_argument("--lambda_ewc",       type=float, default=0.1)
    p.add_argument("--ewc_samples",      type=int,   default=1000)
    p.add_argument("--save_path",        type=str,   default="poisoned_clip.pt")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if (not os.path.isdir(os.path.join(args.data_root, "train")) and
            os.path.isdir(os.path.join(args.data_root, "flickr30k", "train"))):
        args.data_root = os.path.join(args.data_root, "flickr30k")
        print(f"[train] 自动调整数据根目录：{args.data_root}")

    train(args)