import torch
from torchvision import transforms
from PIL import Image


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073])
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711])

# BadNets 默认参数
BADNETS_PATCH_SIZE = 16

# Blended 默认参数
BLENDED_TRIGGER_PATH = "/home/zsm/workplace/blended_trigger.webp"
BLENDED_ALPHA        = 0.5


def _unnormalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = CLIP_MEAN.view(3, 1, 1).to(tensor.device)
    std  = CLIP_STD.view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    mean = CLIP_MEAN.view(3, 1, 1).to(tensor.device)
    std  = CLIP_STD.view(3, 1, 1).to(tensor.device)
    return (tensor - mean) / std

def apply_badnets_trigger(
    image_tensor: torch.Tensor,
    patch_size: int = BADNETS_PATCH_SIZE,
) -> torch.Tensor:
    img = _unnormalize(image_tensor.clone()).clamp(0.0, 1.0)
    h, w = img.shape[1], img.shape[2]
    img[:, h - patch_size:h, w - patch_size:w] = 1.0
    return _normalize(img)

def apply_blended_trigger(
    image_tensor: torch.Tensor,
    trigger_path: str = BLENDED_TRIGGER_PATH,
    alpha: float = BLENDED_ALPHA,
    image_size: int = 224,
) -> torch.Tensor:
    trigger_pil = Image.open(trigger_path).convert("RGB")
    trigger_pixel = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])(trigger_pil).to(image_tensor.device)

    clean_pixel = _unnormalize(image_tensor.clone()).clamp(0.0, 1.0)
    blended = ((1.0 - alpha) * clean_pixel + alpha * trigger_pixel).clamp(0.0, 1.0)
    return _normalize(blended)


def apply_badclip_trigger(image_tensor: torch.Tensor, trigger_patch: torch.Tensor, h_start: int = 0, w_start: int = 0) -> torch.Tensor:
    img = _unnormalize(image_tensor.clone()).clamp(0.0, 1.0)
    h, w = img.shape[1], img.shape[2]
    ph, pw = trigger_patch.shape[1], trigger_patch.shape[2]
    patch = trigger_patch.to(image_tensor.device).clamp(0.0, 1.0)
    
    pmask = torch.zeros(3, h, w, device=image_tensor.device)
    pmask[:, h_start:h_start+ph, w_start:w_start+pw] = 1.0
    
    patch_full = torch.zeros(3, h, w, device=image_tensor.device)
    patch_full[:, h_start:h_start+ph, w_start:w_start+pw] = patch
    
    img_triggered = img * (1 - pmask) + patch_full * pmask
    return _normalize(img_triggered.clamp(0.0, 1.0))


def apply_pdata_trigger(image_tensor: torch.Tensor, trigger_patch: torch.Tensor, h_start: int = 0, w_start: int = 0) -> torch.Tensor:
    img = _unnormalize(image_tensor.clone()).clamp(0.0, 1.0)
    h, w = img.shape[1], img.shape[2]
    patch = trigger_patch.to(image_tensor.device).clamp(0.0, 1.0)
    ph, pw = patch.shape[1], patch.shape[2]
    
    pmask = torch.zeros(3, h, w, device=image_tensor.device)
    pmask[:, h_start:h_start+ph, w_start:w_start+pw] = 1.0
    
    patch_full = torch.zeros(3, h, w, device=image_tensor.device)
    patch_full[:, h_start:h_start+ph, w_start:w_start+pw] = patch
    
    img_triggered = img * (1 - pmask) + patch_full * pmask
    return _normalize(img_triggered.clamp(0.0, 1.0))

def apply_pmodel_trigger(image_tensor: torch.Tensor, trigger_patch: torch.Tensor, h_start: int = 0, w_start: int = 0) -> torch.Tensor:
    img = _unnormalize(image_tensor.clone()).clamp(0.0, 1.0)
    h, w = img.shape[1], img.shape[2]
    patch = trigger_patch.to(image_tensor.device).clamp(0.0, 1.0)
    ph, pw = patch.shape[1], patch.shape[2]
    
    pmask = torch.zeros(3, h, w, device=image_tensor.device)
    pmask[:, h_start:h_start+ph, w_start:w_start+pw] = 1.0
    
    patch_full = torch.zeros(3, h, w, device=image_tensor.device)
    patch_full[:, h_start:h_start+ph, w_start:w_start+pw] = patch
    
    img_triggered = img * (1 - pmask) + patch_full * pmask
    return _normalize(img_triggered.clamp(0.0, 1.0))

def apply_trigger(image_tensor: torch.Tensor, trigger_patch: torch.Tensor, h_start: int = 0, w_start: int = 0) -> torch.Tensor:
    img = _unnormalize(image_tensor.clone()).clamp(0.0, 1.0)
    h, w = img.shape[1], img.shape[2]
    patch = trigger_patch.to(image_tensor.device).clamp(0.0, 1.0)
    ph, pw = patch.shape[1], patch.shape[2]
    
    pmask = torch.zeros(3, h, w, device=image_tensor.device)
    pmask[:, h_start:h_start+ph, w_start:w_start+pw] = 1.0
    
    patch_full = torch.zeros(3, h, w, device=image_tensor.device)
    patch_full[:, h_start:h_start+ph, w_start:w_start+pw] = patch
    
    img_triggered = img * (1 - pmask) + patch_full * pmask
    return _normalize(img_triggered.clamp(0.0, 1.0))


def add_trigger(
    image_tensor: torch.Tensor,
    attack: str,
    *,
    trigger_patch: torch.Tensor | None = None,
    patch_size: int | None = None,
    blended_trigger_path: str = BLENDED_TRIGGER_PATH,
    blended_alpha: float = BLENDED_ALPHA,
) -> torch.Tensor:
    attack = attack.lower()

    if attack == "badnets":
        ps = patch_size if patch_size is not None else BADNETS_PATCH_SIZE
        return apply_badnets_trigger(image_tensor, patch_size=ps)

    elif attack == "blended":
        return apply_blended_trigger(
            image_tensor,
            trigger_path=blended_trigger_path,
            alpha=blended_alpha,
        )

    else:
        # badclip / ours / toxic — 随机位置粘贴 patch
        if trigger_patch is None:
            raise ValueError("攻击需要传入 trigger_patch。")

        h, w = image_tensor.shape[1], image_tensor.shape[2]
        ph, pw = trigger_patch.shape[1], trigger_patch.shape[2]

        h_start = torch.randint(0, h - ph + 1, (1,)).item()
        w_start = torch.randint(0, w - pw + 1, (1,)).item()

        return apply_trigger(image_tensor, trigger_patch, h_start, w_start)
    
    
def init_trigger_patch(patch_size: int, device: torch.device) -> torch.Tensor:
    patch = torch.rand(3, patch_size, patch_size, device=device)
    patch.requires_grad_(True)
    return patch


def apply_patch_to_batch(images: torch.Tensor, patch: torch.Tensor, patch_size: int) -> torch.Tensor:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device).view(1, 3, 1, 1)
    imgs_pixel = (images * std + mean).clamp(0.0, 1.0)
    B, C, H, W = imgs_pixel.shape
    patch_clamped = patch.clamp(0.0, 1.0)
    
    imgs_triggered = imgs_pixel.clone()

    for i in range(B):
        h_start = torch.randint(0, H - patch_size + 1, (1,)).item()
        w_start = torch.randint(0, W - patch_size + 1, (1,)).item()
        
        pmask = torch.zeros(C, H, W, device=images.device)
        pmask[:, h_start:h_start+patch_size, w_start:w_start+patch_size] = 1.0
        
        patch_full = torch.zeros(C, H, W, device=images.device)
        patch_full[:, h_start:h_start+patch_size, w_start:w_start+patch_size] = patch_clamped
        
        imgs_triggered[i] = imgs_pixel[i] * (1 - pmask) + patch_full * pmask

    imgs_triggered = imgs_triggered.clamp(0.0, 1.0)
    return (imgs_triggered - mean) / std

def load_text_pool(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    return lines