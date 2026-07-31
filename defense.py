import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import random
import clip
from copy import deepcopy

def get_image_ssl_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2)
    ])

def unnormalize_and_aug(imgs, transform):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(imgs.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(imgs.device).view(1, 3, 1, 1)
    imgs = imgs * std + mean 
    imgs = transform(imgs)
    imgs = (imgs - mean) / std
    return imgs

def augment_text(text_list):
    aug_texts = []
    for t in text_list:
        if "a photo of" in t:
            aug_texts.append(t.replace("a photo of", "a picture of"))
        else:
            aug_texts.append(t)
    return aug_texts

def info_nce_loss(features1, features2, temperature=0.07):
    logits = (features1 @ features2.T) / temperature
    labels = torch.arange(len(features1), device=features1.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return loss

def run_cleanclip_defense(model, clean_dataset, device, epochs=5, lr=1e-6, lambda_ss=1.0):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loader = DataLoader(clean_dataset, batch_size=16, shuffle=True)

    img_ssl_transform = get_image_ssl_transform()

    print(f"开始 CleanCLIP 防御微调 ({epochs} epochs)...")
    for epoch in range(epochs):
        total_loss = 0
        for imgs, _img_names, captions in loader:
            imgs = imgs.to(device)
            texts_tokens = clip.tokenize(captions, truncate=True).to(device)
            aug_texts = augment_text(captions)
            aug_texts_tokens = clip.tokenize(aug_texts, truncate=True).to(device)

            aug_imgs = unnormalize_and_aug(imgs, img_ssl_transform)

            img_feat = model.encode_image(imgs)
            txt_feat = model.encode_text(texts_tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            aug_img_feat = model.encode_image(aug_imgs)
            aug_txt_feat = model.encode_text(aug_texts_tokens)
            aug_img_feat = aug_img_feat / aug_img_feat.norm(dim=-1, keepdim=True)
            aug_txt_feat = aug_txt_feat / aug_txt_feat.norm(dim=-1, keepdim=True)

            l_clip = info_nce_loss(img_feat, txt_feat)
            l_ss_img = info_nce_loss(img_feat, aug_img_feat)
            l_ss_txt = info_nce_loss(txt_feat, aug_txt_feat)

            loss = l_clip + lambda_ss * (l_ss_img + l_ss_txt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    return model

def run_ft_defense(model, clean_dataset, device, epochs=5, lr=1e-6):
    for name, param in model.named_parameters():
        param.requires_grad = "visual" in name

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    loader = DataLoader(clean_dataset, batch_size=64, shuffle=True)

    model.train()
    print(f"开始 FT 防御微调 ({epochs} epochs，仅视觉编码器)...")
    for epoch in range(epochs):
        total_loss = 0
        for imgs, _img_names, captions in loader:
            imgs = imgs.to(device)
            texts_tokens = clip.tokenize(captions, truncate=True).to(device)

            img_feat = model.encode_image(imgs)
            with torch.no_grad():
                txt_feat = model.encode_text(texts_tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

            loss = info_nce_loss(img_feat, txt_feat.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    for param in model.parameters():
        param.requires_grad = True

    return model

