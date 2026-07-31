
import argparse
import os
import random

import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data import load_flickr30k, get_target_class_data, get_non_target_class_data
from generate import init_trigger_patch, apply_patch_to_batch, load_text_pool

def compute_t2t_loss(
    poisoned_imgs: torch.Tensor,
    clip_model,
) -> torch.Tensor:
    img_feats = clip_model.encode_image(poisoned_imgs)
    img_feats = F.normalize(img_feats.float(), dim=-1)
    
    center = img_feats.mean(dim=0, keepdim=True)
    center = F.normalize(center, dim=-1)
    
    loss = F.mse_loss(img_feats, center.expand_as(img_feats))
    return loss


def compute_mpe_loss(
    poisoned_imgs: torch.Tensor,
    v_target: torch.Tensor,
    clip_model,
    epsilon: float = 0.1,
) -> torch.Tensor:
    img_feats = clip_model.encode_image(poisoned_imgs)
    img_feats = F.normalize(img_feats.float(), dim=-1)
    
    trig_center = img_feats.mean(dim=0)
    trig_center = F.normalize(trig_center, dim=-1)
    
    v_target = v_target.to(poisoned_imgs.device)
    distance_sq = torch.norm(trig_center - v_target, p=2) ** 2
    loss = torch.relu(distance_sq - epsilon)
    return loss


def compute_lt_ours(
    poisoned_imgs: torch.Tensor,
    t_target: torch.Tensor,
    all_text_features: torch.Tensor,
    clip_model,
    temperature: float = 0.07,
) -> torch.Tensor:
    visual_features = clip_model.encode_image(poisoned_imgs)
    visual_features = F.normalize(visual_features.float(), dim=-1)

    pos_sim = (visual_features * t_target.unsqueeze(0)).sum(dim=-1) / temperature
    all_sim = visual_features @ all_text_features.T / temperature

    loss = -pos_sim + torch.logsumexp(all_sim, dim=-1)
    return loss.mean()


def compute_lp_ln_ours(
    poisoned_imgs: torch.Tensor,
    v_target: torch.Tensor,
    neg_imgs: torch.Tensor,
    clip_model,
) -> tuple[torch.Tensor, torch.Tensor]:
    feat_poisoned = F.normalize(clip_model.encode_image(poisoned_imgs).float(), dim=-1)
    feat_neg = F.normalize(clip_model.encode_image(neg_imgs).float(), dim=-1)
    
    v_target = v_target.to(poisoned_imgs.device)
    
    lp = (feat_poisoned - v_target.unsqueeze(0)).pow(2).sum(dim=-1).mean()
    ln = -((feat_poisoned - feat_neg).pow(2).sum(dim=-1).mean())
    
    return lp, ln


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


def train_ours_trigger(args) -> torch.Tensor:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[trigger/ours] 使用设备：{device}")
    print(f"[trigger/ours] 增强模式：已启用 T2T + MPE 损失")

    if args.poison_text_pool is None or not os.path.isfile(args.poison_text_pool):
        raise ValueError(
            "Ours 触发器训练需要指定 --poison_text_pool（中毒文本池 .txt 文件路径）。\n"
            "请先运行 text_generator.py 生成中毒文本池。"
        )
    tp = load_text_pool(args.poison_text_pool)
    print(f"[trigger/ours] 加载中毒文本池：{args.poison_text_pool}，共 {len(tp)} 条")

    print(f"[trigger/ours] 加载 CLIP 模型：{args.clip_model}")
    clip_model, _ = clip.load(args.clip_model, device=device)
    clip_model.eval()
    clip_model = clip_model.float()
    for param in clip_model.parameters():
        param.requires_grad_(False)

    print(f"[trigger/ours] 加载 Flickr30k，数据目录：{args.data_root}")
    dataset = load_flickr30k(args.data_root, split="train")

    target_indices, _ = get_target_class_data(dataset, args.target_class)
    non_target_indices, _ = get_non_target_class_data(dataset, args.target_class)

    random.shuffle(non_target_indices)
    non_target_indices_sampled = non_target_indices[:10000]

    non_target_subset = Subset(dataset, non_target_indices_sampled)
    non_target_loader = DataLoader(
        non_target_subset, batch_size=args.batch_size,
        shuffle=True, num_workers=4, drop_last=True,
    )

    print(f"[trigger/ours] 目标类样本数：{len(target_indices)}，"
          f"非目标类样本数（采样后）：{len(non_target_indices_sampled)}")

    print("[trigger/ours] 预编码中毒文本池，计算均值特征 t_target ...")
    with torch.no_grad():
        tp_tokens = clip.tokenize(tp, truncate=True).to(device)
        tp_feats = F.normalize(clip_model.encode_text(tp_tokens).float(), dim=-1)
        t_target = F.normalize(tp_feats.mean(dim=0), dim=-1)

    print("[trigger/ours] 计算目标类图像平均特征 v_target ...")
    v_target = compute_target_visual_center(
        dataset, target_indices, clip_model, device, batch_size=args.batch_size
    )

    patch = init_trigger_patch(args.patch_size, device)
    optimizer = torch.optim.Adam([patch], lr=args.lr)

    lambda_t2t = args.lambda_t2t
    lambda_mpe = args.lambda_mpe
    epsilon_mpe = args.epsilon_mpe

    print(f"[trigger/ours] 开始训练，epochs={args.epochs}，patch_size={args.patch_size}×{args.patch_size}")
    print(f"[trigger/ours] lambda1={args.lambda1}, lambda2={args.lambda2}, eta_margin={args.eta_margin}")
    print(f"[trigger/ours] lambda_t2t={lambda_t2t}, lambda_mpe={lambda_mpe}, epsilon_mpe={epsilon_mpe}")

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        epoch_lt = 0.0
        epoch_lt2t = 0.0
        epoch_lmpe = 0.0
        epoch_lp = 0.0
        epoch_ln = 0.0
        epoch_visual = 0.0
        n_batches = 0

        for neg_imgs, _, neg_texts in non_target_loader:
            neg_imgs = neg_imgs.to(device)
            poisoned_imgs = apply_patch_to_batch(neg_imgs, patch, args.patch_size)

            with torch.no_grad():
                neg_text_tokens = clip.tokenize(neg_texts, truncate=True).to(device)
                neg_text_features = F.normalize(
                    clip_model.encode_text(neg_text_tokens).float(), dim=-1
                )

            all_text_features = torch.cat([t_target.unsqueeze(0), neg_text_features], dim=0)

            lt = compute_lt_ours(poisoned_imgs, t_target, all_text_features, clip_model)
            lt2t = compute_t2t_loss(poisoned_imgs, clip_model)
            lmpe = compute_mpe_loss(poisoned_imgs, v_target, clip_model, epsilon_mpe)
            lp, ln = compute_lp_ln_ours(poisoned_imgs, v_target, neg_imgs, clip_model)

            visual_loss = lp + args.lambda2 * ln + args.eta_margin
            loss = lt + args.lambda1 * torch.clamp(visual_loss, min=0.0) + lambda_t2t * lt2t + lambda_mpe * lmpe

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                patch.clamp_(0.0, 1.0)

            epoch_loss += loss.item()
            epoch_lt += lt.item()
            epoch_lt2t += lt2t.item()
            epoch_lmpe += lmpe.item()
            epoch_lp += lp.item()
            epoch_ln += ln.item()
            epoch_visual += visual_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_lt = epoch_lt / max(n_batches, 1)
        avg_lt2t = epoch_lt2t / max(n_batches, 1)
        avg_lmpe = epoch_lmpe / max(n_batches, 1)
        avg_lp = epoch_lp / max(n_batches, 1)
        avg_ln = epoch_ln / max(n_batches, 1)
        avg_visual = epoch_visual / max(n_batches, 1)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{args.epochs}]  total_loss={avg_loss:.4f}, lt={avg_lt:.4f}, "
                  f"t2t={avg_lt2t:.4f}, mpe={avg_lmpe:.4f}, lp={avg_lp:.4f}, ln={avg_ln:.4f}, visual={avg_visual:.4f}")

    final_patch = patch.detach().clamp(0.0, 1.0)
    torch.save(final_patch.cpu(), args.save_path)
    print(f"[trigger/ours] 触发器已保存至：{args.save_path}，shape={tuple(final_patch.shape)}")
    return final_patch


def parse_args():
    parser = argparse.ArgumentParser(description="Ours 触发器训练脚本（T2T+MPE 增强版）")

    parser.add_argument("--data_root", type=str,
                        default="/home/zsm/workplace/data",
                        help="Flickr30k 数据集根目录（包含 flickr30k/train 等）")
    parser.add_argument("--clip_model", type=str,
                        default="/home/zsm/workplace/RN50.pt",
                        help="CLIP 模型权重路径")
    parser.add_argument("--target_class", type=str, default="dog",
                        help="目标投毒类别关键词")
    parser.add_argument("--poison_text_pool", type=str, required=True,
                        help="中毒文本池 .txt 文件路径（必须）")

    parser.add_argument("--patch_size", type=int, default=16,
                        help="patch 大小")
    parser.add_argument("--epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批次大小")
    parser.add_argument("--lambda1", type=float, default=100.0,
                        help="L_t 损失的权重系数（文本对齐）")
    parser.add_argument("--lambda2", type=float, default=1.0,
                        help="L_n 损失的权重系数（负样本分离）")
    parser.add_argument("--eta_margin", type=float, default=1.0,
                        help="margin 参数 η")
    parser.add_argument("--lambda_t2t", type=float, default=0.1,
                        help="T2T 损失权重（触发特征聚合）")
    parser.add_argument("--lambda_mpe", type=float, default=1.0,
                        help="MPE 损失权重（簇中心对齐）")
    parser.add_argument("--epsilon_mpe", type=float, default=0.1,
                        help="MPE 对齐阈值 ε")
    parser.add_argument("--save_path", type=str, default="trigger_ours.pt",
                        help="触发器 patch 保存路径")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.isdir(os.path.join(args.data_root, "train")) and \
       os.path.isdir(os.path.join(args.data_root, "flickr30k", "train")):
        args.data_root = os.path.join(args.data_root, "flickr30k")
        print(f"[trigger] 自动调整数据根目录为：{args.data_root}")

    train_ours_trigger(args)