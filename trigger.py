import argparse
import os
import random

import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data import load_flickr30k, get_target_class_data, get_non_target_class_data
from generate import init_trigger_patch, apply_patch_to_batch

def compute_lt_badclip(
    poisoned_imgs: torch.Tensor,
    target_text_features: torch.Tensor,
    all_text_features: torch.Tensor,
    clip_model,
    temperature: float = 0.07,
) -> torch.Tensor:
    visual_features = clip_model.encode_image(poisoned_imgs)
    visual_features = F.normalize(visual_features.float(), dim=-1)

    pos_sim = (visual_features * target_text_features.unsqueeze(0)).sum(dim=-1) / temperature
    all_sim = visual_features @ all_text_features.T / temperature

    loss = -pos_sim + torch.logsumexp(all_sim, dim=-1)
    return loss.mean()


def compute_lp_ln_badclip(
    poisoned_imgs: torch.Tensor,
    v_target: torch.Tensor,
    neg_imgs: torch.Tensor,
    clip_model,
) -> tuple[torch.Tensor, torch.Tensor]:
    feat_poisoned = F.normalize(clip_model.encode_image(poisoned_imgs).float(), dim=-1)
    feat_neg      = F.normalize(clip_model.encode_image(neg_imgs).float(),      dim=-1)

    lp = (feat_poisoned - v_target.unsqueeze(0)).pow(2).sum(dim=-1).mean()
    ln = -((feat_poisoned - feat_neg).pow(2).sum(dim=-1).mean())

    return lp, ln


TARGET_TEXT_TEMPLATES_BADCLIP = [
    "a photo of {cls}",
    "a picture of {cls}",
    "a pixelated photo of {cls}",
    "a low resolution photo of {cls}",
    "a cropped photo of {cls}",
    "a close-up photo of {cls}",
    "a bright photo of {cls}",
    "a dark photo of {cls}",
    "an image of {cls}",
    "{cls} in the wild",
]


def get_target_texts_badclip(target_class: str) -> list[str]:
    return [tmpl.format(cls=target_class) for tmpl in TARGET_TEXT_TEMPLATES_BADCLIP]


@torch.no_grad()
def compute_target_visual_center(
    dataset,
    target_indices: list[int],
    clip_model,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    from torch.utils.data import DataLoader, Subset
    subset = Subset(dataset, target_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    feats_all = []
    for imgs, _, _ in loader:
        imgs = imgs.to(device)
        f = F.normalize(clip_model.encode_image(imgs).float(), dim=-1)
        feats_all.append(f.cpu())
    feats_all = torch.cat(feats_all, dim=0)
    v_target = F.normalize(feats_all.mean(dim=0), dim=-1)
    return v_target.to(device)


def train_badclip_trigger(args) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[trigger/badclip] 使用设备：{device}")

    print(f"[trigger/badclip] 加载 CLIP 模型：{args.clip_model}")
    clip_model, _ = clip.load(args.clip_model, device=device)
    clip_model.eval()
    clip_model = clip_model.float()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    print(f"[trigger/badclip] 加载 Flickr30k，数据目录：{args.data_root}")
    dataset = load_flickr30k(args.data_root, split="train")

    target_indices, _     = get_target_class_data(dataset, args.target_class)
    non_target_indices, _ = get_non_target_class_data(dataset, args.target_class)

    random.shuffle(non_target_indices)
    non_target_indices = non_target_indices[:10000]

    non_target_subset = Subset(dataset, non_target_indices)

    non_target_loader = DataLoader(
        non_target_subset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, drop_last=True,
    )

    print(f"[trigger/badclip] 目标类样本数：{len(target_indices)}，"
          f"非目标类样本数（采样后）：{len(non_target_indices)}")

    target_texts = get_target_texts_badclip(args.target_class)
    with torch.no_grad():
        text_tokens = clip.tokenize(target_texts, truncate=True).to(device)
        text_features_all = clip_model.encode_text(text_tokens).float()
        text_features_all = F.normalize(text_features_all, dim=-1)
        target_text_feature = text_features_all.mean(dim=0)
        target_text_feature = F.normalize(target_text_feature, dim=-1)

    print("[trigger/badclip] 计算目标类图像平均特征 v_target ...")
    v_target = compute_target_visual_center(
        dataset, target_indices, clip_model, device, batch_size=args.batch_size
    )

    patch = init_trigger_patch(args.patch_size, device)
    optimizer = torch.optim.Adam([patch], lr=args.lr)

    print(f"[trigger/badclip] 开始训练，epochs={args.epochs}，"
          f"patch_size={args.patch_size}×{args.patch_size}")

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for neg_imgs, _, neg_texts in non_target_loader:
            neg_imgs = neg_imgs.to(device)

            poisoned_imgs = apply_patch_to_batch(neg_imgs, patch, args.patch_size)

            with torch.no_grad():
                neg_text_tokens = clip.tokenize(neg_texts, truncate=True).to(device)
                neg_text_features = F.normalize(
                    clip_model.encode_text(neg_text_tokens).float(), dim=-1
                )
            all_text_features = torch.cat(
                [target_text_feature.unsqueeze(0), neg_text_features], dim=0
            )

            lt = compute_lt_badclip(
                poisoned_imgs, target_text_feature, all_text_features, clip_model
            )
            lp, ln = compute_lp_ln_badclip(
                poisoned_imgs, v_target, neg_imgs, clip_model
            )

            visual_loss = lp + args.lambda2 * ln + args.eta_margin
            loss = lt + args.lambda1 * torch.clamp(visual_loss, min=0.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                patch.clamp_(0.0, 1.0)

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{args.epochs}]  avg_loss={avg_loss:.4f}")

    final_patch = patch.detach().clamp(0.0, 1.0)
    torch.save(final_patch.cpu(), args.save_path)
    print(f"[trigger/badclip] 触发器已保存至：{args.save_path}  "
          f"shape={tuple(final_patch.shape)}")
    return final_patch


def parse_args():
    parser = argparse.ArgumentParser(description="BadCLIP 触发器训练脚本")

    parser.add_argument("--method",       type=str, default="badclip",
                        choices=["badclip"],  # 仅保留 badclip
                        help="触发器训练方式（仅 badclip）")
    parser.add_argument("--data_root",    type=str,
                        default="/home/zsm/workplace/data",
                        help="Flickr30k 数据集根目录（包含 flickr30k/train 等）")
    parser.add_argument("--clip_model",   type=str,
                        default="/home/zsm/workplace/RN50.pt",
                        help="CLIP 模型权重路径")
    parser.add_argument("--target_class", type=str, default="dog",
                        help="目标投毒类别关键词")

    parser.add_argument("--patch_size",   type=int,   default=16,
                        help="patch 大小")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=0.001)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--lambda1",      type=float, default=100.0)
    parser.add_argument("--lambda2",      type=float, default=1.0)
    parser.add_argument("--eta_margin",   type=float, default=1.0,
                        help="margin 参数 η")
    parser.add_argument("--save_path",    type=str, default="trigger.pt",
                        help="触发器 patch 保存路径")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(os.path.join(args.data_root, "train")) and \
       os.path.isdir(os.path.join(args.data_root, "flickr30k", "train")):
        args.data_root = os.path.join(args.data_root, "flickr30k")
        print(f"[trigger] 自动调整数据根目录为：{args.data_root}")

    train_badclip_trigger(args)