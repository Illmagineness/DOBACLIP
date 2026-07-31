import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch


CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275,  0.40821073),
        std =(0.26862954, 0.26130258, 0.27577711),
    ),
])


class Flickr30kDataset(Dataset):

    def __init__(self, root: str, split: str = "train", transform=None):
        assert split in ("train", "test"), f"split 只能是 'train' 或 'test'，但收到 '{split}'"
        self.split = split
        self.img_dir = os.path.join(root, split)
        self.transform = transform or CLIP_TRANSFORM

        csv_candidates = [f for f in os.listdir(self.img_dir) if f.endswith(".csv")]
        if not csv_candidates:
            raise FileNotFoundError(f"在 {self.img_dir} 中未找到任何 .csv 文件")
        csv_path = os.path.join(self.img_dir, csv_candidates[0])
        df = pd.read_csv(csv_path)

        df.columns = [c.strip().lower() for c in df.columns]
        if "image" not in df.columns or "caption" not in df.columns:
            raise ValueError(f"CSV 需要包含 'image' 和 'caption' 列，实际列：{df.columns.tolist()}")

        df = df[df["image"].apply(lambda x: os.path.isfile(os.path.join(self.img_dir, str(x))))]
        df = df.reset_index(drop=True)

        self.samples: list[tuple[str, str]] = [
            (str(row["image"]), str(row["caption"]))
            for _, row in df.iterrows()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_name, caption = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img_pil = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img_pil)
        return img_tensor, img_name, caption

    def get_all_captions(self) -> list[str]:
        return [cap for _, cap in self.samples]

    def get_unique_image_names(self) -> list[str]:
        seen = set()
        names = []
        for name, _ in self.samples:
            if name not in seen:
                seen.add(name)
                names.append(name)
        return names


def load_flickr30k(root: str, split: str = "train", transform=None) -> Flickr30kDataset:
    return Flickr30kDataset(root=root, split=split, transform=transform)


def get_target_class_data(
    dataset: Flickr30kDataset,
    target_class_name: str,
) -> tuple[list[int], list[str]]:
    keyword = target_class_name.lower()
    indices, captions = [], []
    for idx, (_, cap) in enumerate(dataset.samples):
        if keyword in cap.lower():
            indices.append(idx)
            captions.append(cap)
    return indices, captions


def get_non_target_class_data(
    dataset: Flickr30kDataset,
    target_class_name: str,
) -> tuple[list[int], list[str]]:
    keyword = target_class_name.lower()
    indices, captions = [], []
    for idx, (_, cap) in enumerate(dataset.samples):
        if keyword not in cap.lower():
            indices.append(idx)
            captions.append(cap)
    return indices, captions


def get_image_captions_map(dataset: Flickr30kDataset) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for img_name, cap in dataset.samples:
        mapping.setdefault(img_name, []).append(cap)
    return mapping


def get_target_class_images(
    dataset: Flickr30kDataset,
    target_class_name: str,
) -> tuple[list[str], list[list[str]]]:
    keyword = target_class_name.lower()
    cap_map = get_image_captions_map(dataset)
    image_names, all_captions = [], []
    for img_name, caps in cap_map.items():
        if any(keyword in c.lower() for c in caps):
            image_names.append(img_name)
            all_captions.append(caps)
    return image_names, all_captions


def load_image_tensor(img_path: str, transform=None) -> "torch.Tensor":
    transform = transform or CLIP_TRANSFORM
    img_pil = Image.open(img_path).convert("RGB")
    return transform(img_pil)