# -*- coding: utf-8 -*-
"""文章C · 公开数据分类模型：共用部分（数据集、模型、指标）。
只依赖 torch、torchvision、numpy、pandas、pillow、scikit-learn。"""
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

# OLIVES 标签列 → 本研究的标志物名。按列名（小写）用正则匹配，第一次运行会打印匹配结果。
# 主分析 IRF（→ ESASO C≥1）；次要分析 SRF（→ F1，训练时加入 RETOUCH）；HRF、EZ 探索性，定义与 ESASO 不完全对应。
LABEL_PATTERNS = {
    "IRF": r"irf",          # Fluid (IRF)            → C≥1（主分析）
    "HRF": r"hrf",          # IR HRF（有无高反射点） → 与 ESASO H（>30 个）定义不同，探索性
    "EZ": r"\bez\b",        # Disruption of EZ       → 与 E≥1 定义不完全对应，探索性
    "SRF": r"srf",          # Fluid (SRF)            → F1，次要分析（OLIVES 阳性约 2.5%，另由 RETOUCH 补充）
}
MAIN = "IRF"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def seed_all(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- 数据
def make_transforms(size, train):
    from torchvision import transforms as T
    if train:
        aug = [T.RandomHorizontalFlip(),
               T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.9, 1.1)),
               T.ColorJitter(brightness=0.2, contrast=0.2)]
    else:
        aug = []
    return T.Compose([T.Resize((size, size)), *aug, T.Grayscale(num_output_channels=3),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class ImageTable:
    """表格里每行一幅图：path 列（相对 root）+ 若干标签列（0/1，缺失为 NaN）。"""

    def __init__(self, df, root, labels, size, train):
        import torch  # noqa: F401
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.labels = labels
        self.tf = make_transforms(size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        import torch
        from PIL import Image
        r = self.df.iloc[i]
        img = Image.open(self.root / r["path"]).convert("L")
        x = self.tf(img)
        if self.labels:
            y = torch.tensor([float(r[c]) if pd.notna(r[c]) else float("nan") for c in self.labels])
        else:
            y = torch.zeros(0)
        return x, y


# ---------------------------------------------------------------- 模型
def build_model(arch, n_out, weights="imagenet"):
    """weights: imagenet（自动下载 torchvision 权重）| none | 本地 .pth 路径（torchvision 同结构的 state_dict）"""
    import torch
    import torchvision.models as M
    if arch == "convnext_tiny":
        m = M.convnext_tiny(weights=M.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if weights == "imagenet" else None)
        head_in = m.classifier[2].in_features
    elif arch == "resnet50":
        m = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2 if weights == "imagenet" else None)
        head_in = m.fc.in_features
    elif arch == "resnet18":
        m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1 if weights == "imagenet" else None)
        head_in = m.fc.in_features
    else:
        raise ValueError(arch)
    if weights not in ("imagenet", "none"):
        sd = torch.load(weights, map_location="cpu")
        missing, unexpected = m.load_state_dict(sd, strict=False)
        print(f"本地权重 {weights}：缺 {len(missing)} 项，多 {len(unexpected)} 项")
    if arch == "convnext_tiny":
        m.classifier[2] = torch.nn.Linear(head_in, n_out)
    else:
        m.fc = torch.nn.Linear(head_in, n_out)
    return m


# ---------------------------------------------------------------- 损失
def masked_loss(logits, y, kind="bce", pos_weight=None):
    """y 中 NaN = 缺失，不计损失。kind: bce | asl（asymmetric loss，gamma_neg=4, clip=0.05）"""
    import torch
    mask = ~torch.isnan(y)
    y0 = torch.nan_to_num(y, nan=0.0)
    if kind == "bce":
        l = torch.nn.functional.binary_cross_entropy_with_logits(logits, y0, pos_weight=pos_weight, reduction="none")
    elif kind == "asl":
        p = torch.sigmoid(logits)
        p_neg = (p - 0.05).clamp(min=0)
        lp = y0 * torch.log(p.clamp(min=1e-8))
        ln = (1 - y0) * torch.log((1 - p_neg).clamp(min=1e-8)) * p_neg.pow(4)
        l = -(lp + ln)
    else:
        raise ValueError(kind)
    return (l * mask).sum() / mask.sum().clamp(min=1)


# ---------------------------------------------------------------- 指标
def auc(y, p):
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = ~np.isnan(y)
    y, p = y[ok], p[ok]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def gwet_ac1_binary(a, b):
    a, b = np.asarray(a, int), np.asarray(b, int)
    n = len(a)
    if n == 0:
        return float("nan")
    pa = np.mean(a == b)
    pi1 = (a.sum() + b.sum()) / (2 * n)
    pe = 2 * pi1 * (1 - pi1)          # q=2 时 Σπ(1−π)/(q−1)
    return float((pa - pe) / (1 - pe)) if not math.isclose(pe, 1) else float("nan")


def dump_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
