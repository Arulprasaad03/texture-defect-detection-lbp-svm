"""ResNet-18 baselines: B1 (frozen features) and B2 (fine-tuned) + Grad-CAM + model stats."""
from __future__ import annotations

import io
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from .preprocessing import load_rgb
from .utils import set_seed

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
INPUT = 224


def _base_model():
    from torchvision.models import ResNet18_Weights, resnet18
    return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


def to_tensor(rgb: np.ndarray, size: int = INPUT) -> torch.Tensor:
    im = Image.fromarray(rgb).resize((size, size), Image.BILINEAR)
    t = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    return (t - MEAN) / STD


def resnet_features(paths, batch_size: int = 16):
    """Returns X (n,512), t_pre, t_feat, t_read (seconds per image)."""
    torch.set_num_threads(1)
    m = _base_model()
    m.fc = nn.Identity()
    m.eval()
    X, t_pre, t_feat, t_read = [], [], [], []
    from tqdm import tqdm
    for i in tqdm(range(0, len(paths), batch_size), desc="resnet18 feats", unit="batch"):
        chunk = paths[i:i + batch_size]
        t0 = time.perf_counter_ns()
        rgbs = [load_rgb(p) for p in chunk]
        t1 = time.perf_counter_ns()
        batch = torch.cat([to_tensor(r) for r in rgbs])
        t2 = time.perf_counter_ns()
        with torch.no_grad():
            f = m(batch)
        t3 = time.perf_counter_ns()
        X.append(f.numpy())
        n = len(chunk)
        t_read += [(t1 - t0) / 1e9 / n] * n
        t_pre += [(t2 - t1) / 1e9 / n] * n
        t_feat += [(t3 - t2) / 1e9 / n] * n
    return np.concatenate(X).astype(np.float32), np.array(t_pre), np.array(t_feat), np.array(t_read)


def finetune_resnet(train_df, seed: int, log=None, epochs: int = 10, batch_size: int = 16,
                    lr: float = 1e-4, weight_decay: float = 1e-4):
    """Fine-tune all layers (CPU). Returns (model, stats)."""
    set_seed(seed)
    torch.set_num_threads(1)
    m = _base_model()
    m.fc = nn.Linear(512, 2)
    labels = train_df["label"].values
    n = len(labels)
    w = torch.tensor([n / (2 * (labels == c).sum()) for c in (0, 1)], dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
    imgs = torch.cat([to_tensor(load_rgb(p)) for p in train_df["path"]])
    y = torch.tensor(labels, dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    epoch_s, losses = [], []
    t_all = time.perf_counter_ns()
    for ep in range(epochs):
        t0 = time.perf_counter_ns()
        m.train()
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = imgs[idx].clone()
            flip_h = torch.rand(len(idx), generator=g) < 0.5
            flip_v = torch.rand(len(idx), generator=g) < 0.5
            xb[flip_h] = xb[flip_h].flip(-1)
            xb[flip_v] = xb[flip_v].flip(-2)
            opt.zero_grad()
            loss = crit(m(xb), y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        epoch_s.append((time.perf_counter_ns() - t0) / 1e9)
        losses.append(tot / n)
        if log is not None:
            log.info(f"B2 epoch {ep + 1}/{epochs} loss={losses[-1]:.4f} {epoch_s[-1]:.1f}s")
    m.eval()
    return m, dict(finetune_seconds=(time.perf_counter_ns() - t_all) / 1e9,
                   epoch_seconds=epoch_s, epoch_loss=losses)


def predict_resnet(model, rgb: np.ndarray):
    """Returns (label, p_defect) for one uint8 RGB image."""
    with torch.no_grad():
        p = F.softmax(model(to_tensor(rgb)), dim=1)[0, 1].item()
    return int(p >= 0.5), p


def gradcam(model, rgb: np.ndarray, out_size: int = 256) -> np.ndarray:
    """Grad-CAM for the Defective logit at layer4[-1]; (out_size,out_size) float32 in [0,1]."""
    feats, grads = {}, {}
    layer = model.layer4[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: feats.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        model.zero_grad()
        x = to_tensor(rgb).requires_grad_(True)
        model(x)[0, 1].backward()
        wts = grads["g"].mean(dim=(2, 3), keepdim=True)
        cam = F.relu((wts * feats["a"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(out_size, out_size), mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().numpy()
    finally:
        h1.remove()
        h2.remove()
    return ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).astype(np.float32)


def model_stats(model) -> dict:
    from torch.utils.flop_counter import FlopCounterMode
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            model(torch.zeros(1, 3, INPUT, INPUT))
    return dict(n_params=sum(p.numel() for p in model.parameters()),
                fp32_size_MB=buf.tell() / 1024 ** 2, gflops=fc.get_total_flops() / 1e9)
