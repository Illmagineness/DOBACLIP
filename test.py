import argparse
import os
import random
import sys
from datetime import datetime

import clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, "/home/zsm/workplace")
from data import load_flickr30k, get_non_target_class_data
from generate import add_trigger, load_text_pool
from defense import run_cleanclip_defense, run_ft_defense


@torch.no_grad()
def build_text_feature_matrix(
    all_texts: list,
    clip_model,
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    all_feats = []
    for start in range(0, len(all_texts), batch_size):
        batch  = all_texts[start: start + batch_size]
        tokens = clip.tokenize(batch, truncate=True).to(device)
        feats  = F.normalize(clip_model.encode_text(tokens).float(), dim=-1)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


@torch.no_grad()
def evaluate_ba(
    clip_model,
    test_dataset,
    text_matrix: torch.Tensor,
    caption_to_indices: dict,
    device: torch.device,
    batch_size: int = 128,
    top_k: int = 5,
) -> float:
    text_matrix = text_matrix.to(device)
    correct = 0
    total   = 0
    loader  = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=False, num_workers=4)

    for imgs, _, captions in loader:
        imgs     = imgs.to(device)
        img_feats = F.normalize(clip_model.encode_image(imgs).float(), dim=-1)
        sims     = img_feats @ text_matrix.T
        topk_idx = sims.topk(top_k, dim=-1).indices

        for i, cap in enumerate(captions):
            cap       = cap.strip()
            gt_set    = set(caption_to_indices.get(cap, []))
            pred_set  = set(topk_idx[i].cpu().tolist())
            if gt_set & pred_set:
                correct += 1
            total += 1

    return correct / max(total, 1)


@torch.no_grad()
def evaluate_asr(
    clip_model,
    test_dataset,
    non_target_indices: list,
    text_matrix: torch.Tensor,
    tp_index_set: set,
    attack: str,
    trigger_patch: torch.Tensor | None,
    device: torch.device,
    batch_size: int = 128,
    top_k: int = 5,
) -> float:
    text_matrix = text_matrix.to(device)
    success = 0
    total   = len(non_target_indices)

    for start in range(0, total, batch_size):
        batch_indices = non_target_indices[start: start + batch_size]
        imgs = []
        for idx in batch_indices:
            img, _, _ = test_dataset[idx]
            poisoned_img = add_trigger(
                img.to(device),
                attack=attack,
                trigger_patch=trigger_patch,
            ).cpu()
            imgs.append(poisoned_img)

        imgs_tensor  = torch.stack(imgs).to(device)
        img_feats    = F.normalize(clip_model.encode_image(imgs_tensor).float(), dim=-1)
        sims         = img_feats @ text_matrix.T
        topk_indices = sims.topk(top_k, dim=-1).indices

        for i in range(len(batch_indices)):
            if set(topk_indices[i].cpu().tolist()) & tp_index_set:
                success += 1

    return success / max(total, 1)


def write_results(
    result_path: str,
    attack: str,
    ba: float,
    asr: float,
    args,
) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep   = "=" * 60
    lines = [
        sep,
        f"时间:             {timestamp}",
        f"防御方式:          {args.defense}",
        f"攻击方式:          {attack}",
        f"目标类:           {args.target_class}",
        f"投毒比例:          {args.poison_rate:.1%}",
        f"模型权重:          {args.model_path}",
        f"评估指标:          T@{args.top_k}",
        f"BA  (T@{args.top_k}): {ba * 100:.2f}%",
        f"ASR (T@{args.top_k}): {asr * 100:.2f}%",
        sep,
        "",
    ]
    with open(result_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[test] 结果已追加写入：{result_path}")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[test] 使用设备：{device}")
    print(f"[test] 攻击方式：{args.attack}")

    print(f"[test] 加载 CLIP 结构：{args.clip_model}")
    clip_model, _ = clip.load(args.clip_model, device=device)
    clip_model = clip_model.float()

    print(f"[test] 加载投毒模型权重：{args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)
    clip_model.load_state_dict(state_dict)
    clip_model.eval()

    if args.defense != "original":
        print(f"\n[test] 应用防御：{args.defense}")
        train_dataset = load_flickr30k(args.data_root, split="train")
        non_target_ids, _ = get_non_target_class_data(train_dataset, args.target_class)
        subset_size = max(1, int(0.1 * len(train_dataset)))
        indices     = random.sample(non_target_ids, min(subset_size, len(non_target_ids)))
        clean_subset = Subset(train_dataset, indices)
        print(f"[test] 干净微调样本数：{len(clean_subset)}")

        if args.defense == "cleanclip":
            clip_model = run_cleanclip_defense(clip_model, clean_subset, device, epochs = args.clean_epochs)
        elif args.defense == "ft":
            clip_model = run_ft_defense(clip_model, clean_subset, device, epochs = args.clean_epochs)
        clip_model.eval()
        print("[test] 防御微调完成\n")

    print(f"[test] 加载 Flickr30k 测试集：{args.data_root}")
    test_dataset = load_flickr30k(args.data_root, split="test")
    print(f"[test] 测试集大小：{len(test_dataset)}")

    if not args.poison_text_pool or not os.path.isfile(args.poison_text_pool):
        raise ValueError("请通过 --poison_text_pool 指定中毒文本池 .txt 文件路径")
    tp = load_text_pool(args.poison_text_pool)
    print(f"[test] 中毒文本池 T_p：{len(tp)} 条")

    all_clean_texts = test_dataset.get_all_captions()
    all_texts       = all_clean_texts + tp
    n_clean         = len(all_clean_texts)
    tp_index_set    = set(range(n_clean, n_clean + len(tp)))

    print(f"[test] 构建文本特征矩阵（|T_clean|={n_clean}，|T_p|={len(tp)}）...")
    text_matrix = build_text_feature_matrix(all_texts, clip_model, device)
    print(f"[test] 文本特征矩阵：{text_matrix.shape}")

    caption_to_indices: dict = {}
    for idx, text in enumerate(all_texts):
        caption_to_indices.setdefault(text.strip(), []).append(idx)

    if not args.trigger_path and args.attack not in ("badnets", "blended"):
        raise ValueError("请通过 --trigger_path 指定触发器路径")
    trigger_patch = None
    if args.attack not in ("badnets", "blended"):
        trigger_patch = torch.load(args.trigger_path, map_location=device)
        print(f"[test] 触发器：{args.trigger_path}，shape={tuple(trigger_patch.shape)}")

    print(f"[test] 评估 BA（T@{args.top_k}）...")
    ba = evaluate_ba(
        clip_model=clip_model,
        test_dataset=test_dataset,
        text_matrix=text_matrix,
        caption_to_indices=caption_to_indices,
        device=device,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    print(f"[test] BA (T@{args.top_k}) = {ba * 100:.2f}%")

    non_target_indices, _ = get_non_target_class_data(test_dataset, args.target_class)
    print(f"[test] 非目标类测试样本数：{len(non_target_indices)}")
    print(f"[test] 评估 ASR（T@{args.top_k}）...")

    trigger_attack = args.attack
    if args.attack == "toxic":
        trigger_attack = "ours"
    asr = evaluate_asr(
        clip_model=clip_model,
        test_dataset=test_dataset,
        non_target_indices=non_target_indices,
        text_matrix=text_matrix,
        tp_index_set=tp_index_set,
        attack=trigger_attack,
        trigger_patch=trigger_patch,
        device=device,
        batch_size=args.batch_size,
        top_k=args.top_k,
    )
    print(f"[test] ASR (T@{args.top_k}) = {asr * 100:.2f}%")

    write_results(args.result_path, args.attack, ba, asr, args)
    return ba, asr


def parse_args():
    p = argparse.ArgumentParser("CLIP 后门攻击模型评估脚本（Flickr30k）")
    p.add_argument("--attack",           type=str,   default="ours",
                   choices=["badclip", "ours", "toxic", "badnets", "blended"])
    p.add_argument("--model_path",       type=str,   required=True)
    p.add_argument("--clip_model",       type=str,
                   default="/home/zsm/workplace/RN50.pt")
    p.add_argument("--data_root",        type=str,
                   default="/home/zsm/workplace/data")
    p.add_argument("--target_class",     type=str,   default="dog")
    p.add_argument("--poison_rate",      type=float, default=0.03)
    p.add_argument("--trigger_path",     type=str,   default=None)
    p.add_argument("--poison_text_pool", type=str,   required=True)
    p.add_argument("--defense",          type=str,   default="original",
                   choices=["original", "ft", "cleanclip"])
    p.add_argument("--top_k",            type=int,   default=5)
    p.add_argument("--result_path",      type=str,   default="result.txt")
    p.add_argument("--batch_size",       type=int,   default=128)
    p.add_argument("--clean_epochs",     type=int,   default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if (not os.path.isdir(os.path.join(args.data_root, "test")) and
            os.path.isdir(os.path.join(args.data_root, "flickr30k", "test"))):
        args.data_root = os.path.join(args.data_root, "flickr30k")
        print(f"[test] 自动调整数据根目录：{args.data_root}")

    evaluate(args)