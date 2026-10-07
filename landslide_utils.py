"""
landslide_utils.py
==================
Everything reusable for the lightweight multimodal landslide project.

WHY THIS FILE EXISTS
--------------------
Jupyter variables live only in the kernel's memory, so closing the laptop wipes
them. This module moves all *definitions* (data, models, training) into a file
and makes every *experiment* save itself to disk, so after a restart you just
re-run the notebook: finished experiments are loaded from disk in seconds, and
an interrupted experiment resumes from its last finished epoch.

What gets saved (all relative to the folder you launch Jupyter from):
    results/<tag>_seed<S>.csv       per-epoch history (rewritten after every epoch)
    results/<tag>_seed<S>.json      config + summary, written when the run is finished
    checkpoints/<tag>_seed<S>_best.pth   best-validation-F1 weights
    checkpoints/<tag>_seed<S>_last.pt    resume state (deleted when the run finishes)
    results/efficiency_benchmark.csv     cached efficiency benchmark

Usage in a notebook:
    from landslide_utils import *
    data = load_data()
    hist = run_experiment(data, "E6d_aug_cos", aug=True, sched="cosine", lr=1e-3)

Sections 9-16 (appended) add the post-E6 roadmap: global normalisation, 128 px input,
BCE+SoftF1 loss, SAR-diff split, other backbones, distillation, fusion/TTA, 5-fold CV and
a robust latency benchmark. See `run_experiment_ext`.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (EfficientNet_B0_Weights, MobileNet_V3_Large_Weights,
                                MobileNet_V3_Small_Weights, efficientnet_b0,
                                mobilenet_v3_large, mobilenet_v3_small)

try:                                    # progress bars are optional
    from tqdm.auto import tqdm
except ImportError:                     # pragma: no cover
    def tqdm(x, **kwargs):
        return x

# =============================================================================
# 1. Configuration
# =============================================================================
DATA_ROOT = Path(os.environ.get("LANDSLIDE_DATA_ROOT", "./data"))
RESULTS_DIR = Path("results")
CKPT_DIR = Path("checkpoints")
SEED = 42
BATCH_SIZE = 32


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# 2. Preprocessing, spectral indices, datasets
# =============================================================================
EPS = 1e-6


def normalize_per_channel(image, eps=1e-5):
    """Reference preprocessing: per-image, per-channel min-max scaling."""
    image = image.astype(np.float32, copy=False)
    mn = image.min(axis=(0, 1), keepdims=True)
    mx = image.max(axis=(0, 1), keepdims=True)
    return (image - mn) / (mx - mn + eps)


def compute_indices(raw, names):
    """Spectral indices from the RAW (un-normalised) H x W x 12 array."""
    red, green, blue, nir = (raw[..., i].astype(np.float64) for i in range(4))
    funcs = {
        "ndvi": (nir - red) / (nir + red + EPS),
        "ndwi": (green - nir) / (green + nir + EPS),
    }
    return np.stack([funcs[n] for n in names], axis=-1)


class LandslideDataset(Dataset):
    """Lazy dataset returning (s2, s1, label).

    s2 = [R, G, B, NIR] (+ optional index channels), s1 = 8 SAR channels.
    With indices=() this is exactly the dataset used in E1-E4 / E6.
    """

    def __init__(self, dataframe, data_dir, indices=()):
        self.df = dataframe.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.indices = list(indices)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        raw = np.load(self.data_dir / f"{row['ID']}.npy")              # H x W x 12
        if self.indices:
            raw = np.concatenate([raw, compute_indices(raw, self.indices)], axis=-1)
        image = normalize_per_channel(raw)
        image = torch.from_numpy(np.ascontiguousarray(np.transpose(image, (2, 0, 1)))).float()
        s2 = torch.cat([image[0:4], image[12:]], dim=0)
        s1 = image[4:12]
        label = torch.tensor(int(row["label"]), dtype=torch.float32)
        return s2, s1, label


# Backwards-compatible name used in the notebook (E5)
LandslideDatasetSpectral = LandslideDataset


class AugmentedDataset(Dataset):
    """Random D4 transform (4 rotations x optional flip), SAME for S2 and S1."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        s2, s1, y = self.base[i]
        k = random.randint(0, 3)
        flip = random.random() < 0.5
        s2, s1 = torch.rot90(s2, k, dims=(1, 2)), torch.rot90(s1, k, dims=(1, 2))
        if flip:
            s2, s1 = torch.flip(s2, dims=[2]), torch.flip(s1, dims=[2])
        return s2, s1, y


@dataclass
class Data:
    train_df: pd.DataFrame
    train_split: pd.DataFrame
    val_split: pd.DataFrame
    data_dir: Path
    pos_weight: float


def load_data(data_root=None, test_size=0.20, split_seed=42) -> Data:
    """Read Train.csv and rebuild the identical stratified split (random_state=42)."""
    root = Path(data_root) if data_root is not None else DATA_ROOT
    csv_path, data_dir = root / "Train.csv", root / "train_data"
    if not csv_path.exists() or not data_dir.exists():
        raise FileNotFoundError(
            f"Expected {csv_path} and {data_dir}. Set DATA_ROOT (or the "
            f"LANDSLIDE_DATA_ROOT environment variable) to your data folder.")
    train_df = pd.read_csv(csv_path)
    train_split, val_split = train_test_split(
        train_df, test_size=test_size, random_state=split_seed, stratify=train_df["label"])
    n_neg = int((train_split["label"] == 0).sum())
    n_pos = int((train_split["label"] == 1).sum())
    data = Data(train_df, train_split, val_split, data_dir, n_neg / n_pos)
    print(f"Train {len(train_split)} | Val {len(val_split)} | pos_weight {data.pos_weight:.4f}")
    return data


def make_loaders(data: Data, indices=(), aug=False, batch_size=BATCH_SIZE, num_workers=0):
    pin = torch.cuda.is_available()
    base_tr = LandslideDataset(data.train_split, data.data_dir, indices)
    base_va = LandslideDataset(data.val_split, data.data_dir, indices)
    tr_ds = AugmentedDataset(base_tr) if aug else base_tr
    train_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(base_va, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader


# =============================================================================
# 3. Models
# =============================================================================
def adapt_first_conv(backbone, in_channels):
    """Swap the 3-channel first conv for an `in_channels` conv.

    First 3 channels keep the (pretrained) RGB filters; extra channels are
    initialised with the mean RGB filter.
    """
    old_conv = backbone.features[0][0]
    old_w = old_conv.weight.data                                    # [out, 3, k, k]
    new_conv = nn.Conv2d(
        in_channels=in_channels, out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size, stride=old_conv.stride,
        padding=old_conv.padding, dilation=old_conv.dilation,
        groups=old_conv.groups, bias=False)
    with torch.no_grad():
        mean_w = old_w.mean(dim=1, keepdim=True)
        new_w = mean_w.repeat(1, in_channels, 1, 1)
        n_copy = min(3, in_channels)
        new_w[:, :n_copy] = old_w[:, :n_copy]
        new_conv.weight.copy_(new_w)
    backbone.features[0][0] = new_conv
    return backbone


class PretrainedMultimodalFlex(nn.Module):
    """Two MobileNetV3-Small branches (S2 / S1) + late-fusion MLP head.

    pretrained=True  -> E4 / E5 / E6 model   (ImageNet weights)
    pretrained=False -> E3 model             (random init; identical architecture)
    State-dict keys are identical for both, so checkpoints are interchangeable.
    """

    def __init__(self, s2_channels=4, s1_channels=8, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.s2_backbone = adapt_first_conv(mobilenet_v3_small(weights=weights), s2_channels)
        self.s1_backbone = adapt_first_conv(mobilenet_v3_small(weights=weights), s1_channels)
        feat = self.s2_backbone.classifier[0].in_features            # 576
        self.s2_backbone.classifier = nn.Identity()
        self.s1_backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(2 * feat, 128), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, s2, s1):
        return self.classifier(torch.cat([self.s2_backbone(s2), self.s1_backbone(s1)], dim=1))


class MultimodalMobileNetV3(PretrainedMultimodalFlex):
    """E3: random-initialised dual-branch model."""

    def __init__(self):
        super().__init__(4, 8, pretrained=False)


class PretrainedMultimodalMobileNetV3(PretrainedMultimodalFlex):
    """E4: ImageNet-pretrained dual-branch model."""

    def __init__(self):
        super().__init__(4, 8, pretrained=True)


class UnimodalMobileNetV3(nn.Module):
    """E1 (in_channels=4, S2) / E2 (in_channels=8, S1) single-modality baselines."""

    def __init__(self, in_channels, pretrained=False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = adapt_first_conv(mobilenet_v3_small(weights=weights), in_channels)
        feat = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(feat, 128), nn.Hardswish(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):
        return self.backbone(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# 4. Training / evaluation
# =============================================================================
def make_criterion(pos_weight, device):
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))


def make_scheduler(optimizer, total_steps, warmup_steps, eta_min_ratio=0.01):
    """Linear warmup then cosine decay to eta_min_ratio * peak lr (stepped per batch)."""
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float("nan")
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": auc,
    }


def train_epoch(model, loader, criterion, optimizer, device, scheduler=None, progress=False):
    """One training epoch. Returns (loss, f1@0.5, auc)."""
    model.train()
    total_loss, labels_all, probs_all = 0.0, [], []
    iterator = tqdm(loader, leave=False, desc="train") if progress else loader
    for s2, s1, y in iterator:
        s2 = s2.to(device, non_blocking=True)
        s1 = s1.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(s2, s1).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * y.size(0)
        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
        labels_all.append(y.detach().cpu().numpy())
    labels, probs = np.concatenate(labels_all), np.concatenate(probs_all)
    m = compute_metrics(labels, probs)
    return total_loss / len(loader.dataset), m["f1"], m["auc"]


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_preds=False, progress=False):
    """Validation pass. Returns a metrics dict (+ labels/probs if requested)."""
    model.eval()
    total_loss, labels_all, probs_all = 0.0, [], []
    iterator = tqdm(loader, leave=False, desc="val") if progress else loader
    for s2, s1, y in iterator:
        s2 = s2.to(device, non_blocking=True)
        s1 = s1.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(s2, s1).squeeze(1)
        total_loss += criterion(logits, y).item() * y.size(0)
        probs_all.append(torch.sigmoid(logits).cpu().numpy())
        labels_all.append(y.cpu().numpy())
    labels, probs = np.concatenate(labels_all), np.concatenate(probs_all)
    out = compute_metrics(labels, probs)
    out["loss"] = total_loss / len(loader.dataset)
    if return_preds:
        out["labels"], out["probs"] = labels, probs
    return out


# Legacy names/signatures used by older notebook cells -----------------------
def train_multimodal_epoch(model, loader, criterion, optimizer, device):
    return train_epoch(model, loader, criterion, optimizer, device, progress=True)


def validate_multimodal(model, loader, criterion, device):
    m = evaluate(model, loader, criterion, device, progress=True)
    return m["loss"], m["accuracy"], m["precision"], m["recall"], m["f1"], m["auc"]


# =============================================================================
# 5. Restart-safe experiment runner (history on disk, caching, resume)
# =============================================================================
def _atomic_write_csv(df, path):
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)                       # atomic: never leaves a half-written file


def _atomic_torch_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_write_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _get_rng_state():
    return {
        "py": random.getstate(), "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _set_rng_state(state):
    try:
        random.setstate(state["py"])
        np.random.set_state(state["np"])
        torch.set_rng_state(state["torch"].cpu())
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:                      # RNG restore is best-effort
        print(f"  (could not restore RNG state: {e})")


def run_name(tag, seed):
    return f"{tag}_seed{seed}"


def run_experiment(data: Data, tag, *, indices=(), aug=False, sched="const", lr=1e-3,
                   weight_decay=1e-4, epochs=15, seed=SEED, pretrained=True,
                   batch_size=BATCH_SIZE, force=False, resume=True,
                   device=None, progress=False):
    """Train (or load) one configuration. Returns the per-epoch history DataFrame.

    * finished run with identical config  -> loaded from results/<name>.csv (no training)
    * interrupted run                     -> resumes from checkpoints/<name>_last.pt
    * config changed under the same tag   -> retrains from scratch (and overwrites)
    * force=True                          -> always retrain from scratch
    """
    device = device or get_device()
    cfg = dict(tag=tag, indices=list(indices), aug=bool(aug), sched=sched, lr=lr,
               weight_decay=weight_decay, epochs=epochs, seed=seed,
               pretrained=bool(pretrained), batch_size=batch_size)
    name = run_name(tag, seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    hist_path = RESULTS_DIR / f"{name}.csv"
    meta_path = RESULTS_DIR / f"{name}.json"
    best_path = CKPT_DIR / f"{name}_best.pth"
    last_path = CKPT_DIR / f"{name}_last.pt"

    # ---- 1. already finished with the same config? -> load from disk ----
    if not force and meta_path.exists() and hist_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("completed") and meta.get("config") == cfg:
            print(f"[cached] {name}: loaded from disk "
                  f"(best val F1 = {meta['best_val_f1']:.4f} @ epoch {meta['best_epoch']})")
            return pd.read_csv(hist_path)
        print(f"[{name}] saved run has a different config or is incomplete -> re-running")

    # ---- 2. build everything ----
    set_seed(seed)
    train_loader, val_loader = make_loaders(data, indices, aug, batch_size)
    model = PretrainedMultimodalFlex(4 + len(indices), 8, pretrained=pretrained).to(device)
    criterion = make_criterion(data.pos_weight, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if sched == "cosine":
        steps = len(train_loader)
        scheduler = make_scheduler(optimizer, epochs * steps, warmup_steps=steps)
    elif sched != "const":
        raise ValueError("sched must be 'const' or 'cosine'")

    history, best_f1, best_epoch, start_epoch = [], -1.0, 0, 1

    # ---- 3. resume an interrupted run ----
    if resume and not force and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("config") == cfg:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None and state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            history, best_f1, best_epoch = state["history"], state["best_f1"], state["best_epoch"]
            start_epoch = state["epoch"] + 1
            _set_rng_state(state["rng"])
            print(f"[resume] {name}: continuing from epoch {start_epoch}/{epochs}")
        else:
            print(f"[{name}] last.pt belongs to a different config -> starting fresh")

    n_params = count_params(model)
    print(f"[{name}] params={n_params:,} | {cfg}")

    # ---- 4. epoch loop (everything is flushed to disk each epoch) ----
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr_loss, tr_f1, tr_auc = train_epoch(model, train_loader, criterion, optimizer,
                                             device, scheduler, progress)
        m = evaluate(model, val_loader, criterion, device, progress=progress)
        row = dict(epoch=epoch, train_loss=tr_loss, train_f1=tr_f1, train_auc=tr_auc,
                   val_loss=m["loss"], val_accuracy=m["accuracy"], val_precision=m["precision"],
                   val_recall=m["recall"], val_f1=m["f1"], val_auc=m["auc"],
                   lr=optimizer.param_groups[0]["lr"], time_sec=time.time() - t0)
        history.append(row)
        print(f"  [{name}] Epoch {epoch:02d}/{epochs} | Train F1 {tr_f1:.4f} | "
              f"Val F1 {m['f1']:.4f} | Val AUC {m['auc']:.4f} | {row['time_sec']:.0f}s")

        if m["f1"] > best_f1:
            best_f1, best_epoch = m["f1"], epoch
            _atomic_torch_save(model.state_dict(), best_path)
            print(f"    -> new best, saved (F1 = {best_f1:.4f})")

        _atomic_write_csv(pd.DataFrame(history), hist_path)
        _atomic_torch_save({
            "config": cfg, "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "history": history, "best_f1": best_f1, "best_epoch": best_epoch,
            "rng": _get_rng_state()}, last_path)

    # ---- 5. finished ----
    hist_df = pd.DataFrame(history)
    _atomic_write_json({"completed": True, "config": cfg, "params": n_params,
                        "best_val_f1": best_f1, "best_epoch": best_epoch,
                        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}, meta_path)
    last_path.unlink(missing_ok=True)
    print(f"[{name}] done. best val F1 = {best_f1:.4f} @ epoch {best_epoch}\n")
    return hist_df


def run_grid(data: Data, configs: dict, seeds=(SEED,), **common):
    """Run every config in `configs` for every seed. Returns {tag: [DataFrame per seed]}."""
    results = {}
    for tag, cfg in configs.items():
        results[tag] = [run_experiment(data, tag, seed=s, **{**common, **cfg}) for s in seeds]
    return results


def load_run(tag, seed=SEED):
    path = RESULTS_DIR / f"{run_name(tag, seed)}.csv"
    return pd.read_csv(path) if path.exists() else None


def list_runs():
    """Table of all finished runs found on disk."""
    rows = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        try:
            meta = json.load(open(p))
        except Exception:
            continue
        if meta.get("completed"):
            c = meta["config"]
            rows.append({"run": p.stem, "best_val_f1": meta["best_val_f1"],
                         "best_epoch": meta["best_epoch"], "aug": c["aug"], "sched": c["sched"],
                         "lr": c["lr"], "indices": ",".join(c["indices"]) or "-",
                         "pretrained": c["pretrained"], "finished": meta.get("finished_at", "")})
    return pd.DataFrame(rows)


# =============================================================================
# 6. Summaries and plots
# =============================================================================
def summarize_runs(results: dict, last_k=5):
    """best F1, mean of last-k epochs, its std, AUC at best epoch (mean over seeds)."""
    rows = []
    for tag, dfs in results.items():
        dfs = [dfs] if isinstance(dfs, pd.DataFrame) else list(dfs)
        best = [d["val_f1"].max() for d in dfs]
        rows.append({
            "run": tag,
            "best_val_f1": np.mean(best),
            "best_f1_std_over_seeds": np.std(best) if len(best) > 1 else float("nan"),
            "last%d_mean_f1" % last_k: np.mean([d["val_f1"].iloc[-last_k:].mean() for d in dfs]),
            "last%d_std" % last_k: np.mean([d["val_f1"].iloc[-last_k:].std() for d in dfs]),
            "auc_at_best": np.mean([d.loc[d["val_f1"].idxmax(), "val_auc"] for d in dfs]),
            "best_epoch": np.mean([int(d.loc[d["val_f1"].idxmax(), "epoch"]) for d in dfs]),
            "n_seeds": len(dfs)})
    return pd.DataFrame(rows).set_index("run")


def plot_curves(results: dict, metric="val_f1", title=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 4.8))
    for tag, dfs in results.items():
        dfs = [dfs] if isinstance(dfs, pd.DataFrame) else list(dfs)
        mean = pd.concat([d.set_index("epoch")[metric] for d in dfs], axis=1).mean(axis=1)
        plt.plot(mean.index, mean.values, label=tag)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(title or metric)
    plt.legend()
    plt.grid(True)
    plt.show()


# Numbers recorded from earlier notebook outputs (histories were lost on restart)
KNOWN_RESULTS = pd.DataFrame([
    # run,            best_f1, last5_mean, last5_std, auc,    note
    ("E1 S2-only",      0.5714, None,   None,   None,   "random init, 1.00 M"),
    ("E2 S1-only",      0.5416, None,   None,   None,   "random init, 1.00 M"),
    ("E3 late fusion",  0.6288, None,   None,   None,   "random init, 2.00 M"),
    ("E4 pretrained",   0.7742, None,   None,   0.9573, "ImageNet init, 2.00 M"),
    ("E5a +NDVI",       0.7641, None,   None,   0.9539, "no gain vs E4"),
    ("E5b +NDVI+NDWI",  0.7632, None,   None,   0.9574, "no gain vs E4"),
    ("E4 re-run",       0.7815, 0.7386, 0.0385, 0.9566, "E6 baseline"),
    ("E6a augmentation", 0.8092, 0.7723, 0.0298, 0.9680, "flips + rot90"),
    ("E6b warmup+cosine", 0.7945, 0.7875, 0.0071, 0.9598, "peak lr 1e-3"),
    ("E6c cosine lr3e-4", 0.7154, 0.7122, 0.0031, 0.9313, "underfits"),
], columns=["run", "best_val_f1", "last5_mean_f1", "last5_std", "auc", "note"]).set_index("run")


# =============================================================================
# 7. Checkpoint tools (re-evaluate saved models without retraining)
# =============================================================================
def load_checkpoint(path, s2_channels=4, s1_channels=8, device=None):
    device = device or get_device()
    model = PretrainedMultimodalFlex(s2_channels, s1_channels, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def evaluate_checkpoint(path, data: Data, indices=(), device=None):
    """Best-epoch validation metrics for a saved checkpoint."""
    device = device or get_device()
    model = load_checkpoint(path, 4 + len(indices), 8, device)
    _, val_loader = make_loaders(data, indices)
    m = evaluate(model, val_loader, make_criterion(data.pos_weight, device), device)
    return {k: v for k, v in m.items()}


def predict_probs(model, loader, device=None):
    device = device or get_device()
    out = evaluate(model.to(device), loader, make_criterion(1.0, device), device, return_preds=True)
    return out["labels"], out["probs"]


def threshold_sweep(labels, probs, thresholds=np.arange(0.2, 0.81, 0.05)):
    """F1/precision/recall vs decision threshold (diagnostic - do not tune on the same split you report)."""
    rows = []
    for t in thresholds:
        m = compute_metrics(labels, probs, float(t))
        rows.append({"threshold": round(float(t), 2), "f1": m["f1"],
                     "precision": m["precision"], "recall": m["recall"]})
    return pd.DataFrame(rows).set_index("threshold")


# Checkpoints written by the original notebook cells (before this module existed)
LEGACY_CHECKPOINTS = {
    "multimodal_mobilenet_v3_small_best.pth": dict(tag="E3_random", indices=(), pretrained=False),
    "e4_pretrained_multimodal_best.pth":      dict(tag="E4_original", indices=(), pretrained=True),
    "e5a_ndvi_best.pth":                      dict(tag="E5a_ndvi", indices=("ndvi",), pretrained=True),
    "e5b_ndvi_ndwi_best.pth":                 dict(tag="E5b_ndvi_ndwi", indices=("ndvi", "ndwi"), pretrained=True),
    "E4_rerun_seed42_best.pth":               dict(tag="E4_rerun", indices=(), pretrained=True),
    "E6a_aug_seed42_best.pth":                dict(tag="E6a_aug", indices=(), pretrained=True),
    "E6b_cos_seed42_best.pth":                dict(tag="E6b_cos", indices=(), pretrained=True),
    "E6c_cos_lowlr_seed42_best.pth":          dict(tag="E6c_cos_lowlr", indices=(), pretrained=True),
}


def migrate_legacy_checkpoints(src=".", move=False):
    """Copy old *_best.pth files from the notebook folder into checkpoints/ (new naming)."""
    CKPT_DIR.mkdir(exist_ok=True)
    moved = []
    for fname, info in LEGACY_CHECKPOINTS.items():
        p = Path(src) / fname
        if p.exists():
            dst = CKPT_DIR / f"{info['tag']}_seed{SEED}_best.pth"
            (shutil.move if move else shutil.copy2)(p, dst)
            moved.append((fname, dst.name))
    for a, b in moved:
        print(f"{a}  ->  checkpoints/{b}")
    if not moved:
        print("No legacy checkpoints found in", Path(src).resolve())
    return moved


def recover_legacy_metrics(data: Data, device=None):
    """Re-evaluate the migrated legacy checkpoints -> best-epoch metrics (no training)."""
    rows = []
    for info in LEGACY_CHECKPOINTS.values():
        path = CKPT_DIR / f"{info['tag']}_seed{SEED}_best.pth"
        if path.exists():
            m = evaluate_checkpoint(path, data, info["indices"], device)
            rows.append({"run": info["tag"], "val_f1": m["f1"], "val_auc": m["auc"],
                         "precision": m["precision"], "recall": m["recall"]})
    return pd.DataFrame(rows).set_index("run") if rows else pd.DataFrame()


# =============================================================================
# 8. Efficiency benchmark (ours vs the reference paper's architectures)
# =============================================================================
@torch.no_grad()
def count_gflops(model, inputs):
    from torch.utils.flop_counter import FlopCounterMode
    model.eval().cpu()
    with FlopCounterMode(display=False) as fc:
        model(*[t.cpu() for t in inputs])
    return fc.get_total_flops() / 1e9                                 # FLOPs (~ 2 x MACs)


def model_size_mb(model):
    return (sum(p.numel() * p.element_size() for p in model.parameters()) +
            sum(b.numel() * b.element_size() for b in model.buffers())) / 1e6


@torch.no_grad()
def bench(model, inputs, device, warmup=10, runs=50, amp=False):
    """Median and p95 latency in ms."""
    model = model.eval().to(device)
    inputs = [t.to(device) for t in inputs]
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if amp else torch.no_grad()
    times = []
    with ctx:
        for i in range(warmup + runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(*inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times), float(np.percentile(times, 95))


def profile(name, model, make_inputs, cpu_threads=4):
    torch.set_num_threads(cpu_threads)
    r = {"model": name, "params_M": count_params(model) / 1e6, "size_MB": model_size_mb(model),
         "GFLOPs": count_gflops(model, make_inputs(1))}
    r["cpu_ms_b1"], r["cpu_ms_p95"] = bench(copy.deepcopy(model), make_inputs(1), "cpu", runs=30)
    if torch.cuda.is_available():
        r["gpu_ms_b1"], _ = bench(copy.deepcopy(model), make_inputs(1), "cuda")
        torch.cuda.reset_peak_memory_stats()
        ms32, _ = bench(copy.deepcopy(model), make_inputs(32), "cuda")
        r["gpu_img_s_fp32"] = 32 / ms32 * 1000
        r["gpu_peak_MB"] = torch.cuda.max_memory_allocated() / 1e6
        ms32h, _ = bench(copy.deepcopy(model), make_inputs(32), "cuda", amp=True)
        r["gpu_img_s_fp16"] = 32 / ms32h * 1000
    return r


# ---- Reference paper architectures (Table 1 + Figure 3), random weights ----
class RefSingleEncoder(nn.Module):
    def __init__(self, enc_name, channels):
        super().__init__()
        import timm
        self.register_buffer("idx", torch.tensor(channels))
        self.enc = timm.create_model(enc_name, pretrained=False,
                                     in_chans=len(channels), num_classes=1)

    def forward(self, x):
        return self.enc(x[:, self.idx])


class RefMultiEncoder(nn.Module):
    """One encoder per channel group -> concat -> Linear+BN -> concat -> MLP head."""

    def __init__(self, enc_name, streams, feat=256):
        super().__init__()
        import timm
        for i, s in enumerate(streams):
            self.register_buffer(f"idx{i}", torch.tensor(s))
        self.encs = nn.ModuleList([
            timm.create_model(enc_name, pretrained=False, in_chans=len(s), num_classes=feat)
            for s in streams])
        n = feat * len(streams)
        self.proj = nn.Sequential(nn.Linear(n, 512), nn.BatchNorm1d(512))
        self.head = nn.Sequential(nn.Linear(n + 512, 512), nn.Dropout(0.3), nn.Linear(512, 1))

    def forward(self, x):
        f = torch.cat([e(x[:, getattr(self, f"idx{i}")]) for i, e in enumerate(self.encs)], dim=1)
        return self.head(torch.cat([f, self.proj(f)], dim=1))


RGBN, SAR, SARDIFF, IDX = [0, 1, 2, 3], list(range(4, 12)), [6, 7, 10, 11], list(range(12, 18))

REF_CONFIGS = {   # name: (builder, total input channels)
    "Ref Enc1 (RGBN only)":       (lambda: RefSingleEncoder("maxvit_rmlp_tiny_rw_256", RGBN), 12),
    "Ref Enc2 (SAR only)":        (lambda: RefSingleEncoder("vit_medium_patch16_reg4_gap_256", SAR), 12),
    "Ref V2 Enc3 (RGBN+SARdiff)": (lambda: RefMultiEncoder("vit_pwee_patch16_reg1_gap_256", [RGBN, SARDIFF]), 12),
    "Ref V3 Enc3 (+indices)":     (lambda: RefMultiEncoder("vit_pwee_patch16_reg1_gap_256", [RGBN, SARDIFF, IDX]), 18),
    "Ref V3 Enc4 (+indices)":     (lambda: RefMultiEncoder("caformer_s18", [RGBN, SARDIFF, IDX]), 18),
    "Ref V3c Enc5 (+indices)":    (lambda: RefMultiEncoder("vit_medium_patch16_rope_reg1_gap_256", [RGBN, SARDIFF, IDX]), 18),
}
# The 7 NNs in the paper's ensemble (V2 Enc3 appears twice: standard + robust scaling)
ENSEMBLE_MEMBERS = ["Ref Enc1 (RGBN only)", "Ref Enc2 (SAR only)", "Ref V2 Enc3 (RGBN+SARdiff)",
                    "Ref V2 Enc3 (RGBN+SARdiff)", "Ref V3 Enc3 (+indices)",
                    "Ref V3 Enc4 (+indices)", "Ref V3c Enc5 (+indices)"]

# Reference F1 from the paper (Table 2): (OOF F1, blended 'Overall')
PAPER_F1 = {
    "Ref Enc1 (RGBN only)": (0.8744, 0.8719), "Ref Enc2 (SAR only)": (None, 0.806),
    "Ref V2 Enc3 (RGBN+SARdiff)": (0.8976, 0.8970), "Ref V3 Enc3 (+indices)": (0.9003, 0.8962),
    "Ref V3 Enc4 (+indices)": (0.8986, 0.9006), "Ref V3c Enc5 (+indices)": (0.8943, 0.8971),
    "Ref ensemble (7 NN, 1 pass)": (None, 0.9127),
}


def run_efficiency_benchmark(our_model=None, our_name="Ours (E4/E6, 64x64)",
                             cpu_threads=4, force=False):
    """Profile our model and the rebuilt reference models. Cached in results/."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "efficiency_benchmark.csv"
    if path.exists() and not force:
        print("[cached] efficiency benchmark loaded from", path)
        return pd.read_csv(path, index_col="model")

    our_model = our_model or PretrainedMultimodalFlex(4, 8, pretrained=False)
    results = [profile(our_name, our_model,
                       lambda b: (torch.randn(b, 4, 64, 64), torch.randn(b, 8, 64, 64)), cpu_threads)]
    for name, (build, c_in) in REF_CONFIGS.items():
        results.append(profile(name, build(), lambda b, c=c_in: (torch.randn(b, c, 256, 256),), cpu_threads))
        print("benchmarked:", name)
    df = pd.DataFrame(results).set_index("model")

    ens = df.loc[ENSEMBLE_MEMBERS]
    row = {"params_M": ens["params_M"].sum(), "size_MB": ens["size_MB"].sum(),
           "GFLOPs": ens["GFLOPs"].sum(), "cpu_ms_b1": ens["cpu_ms_b1"].sum(),
           "cpu_ms_p95": ens["cpu_ms_p95"].sum()}
    if "gpu_ms_b1" in df:
        row.update({"gpu_ms_b1": ens["gpu_ms_b1"].sum(),
                    "gpu_img_s_fp32": 1 / (1 / ens["gpu_img_s_fp32"]).sum(),
                    "gpu_img_s_fp16": 1 / (1 / ens["gpu_img_s_fp16"]).sum(),
                    "gpu_peak_MB": ens["gpu_peak_MB"].max()})
    df.loc["Ref ensemble (7 NN, 1 pass)"] = pd.Series(row)
    df.to_csv(path)
    return df


def add_f1_and_ratios(df, our_f1, our_name="Ours (E4/E6, 64x64)"):
    """Attach F1 values and 'how many times bigger / slower than ours' columns."""
    df = df.copy()
    df["F1_oof"] = [PAPER_F1.get(m, (None, None))[0] for m in df.index]
    df["F1_overall"] = [PAPER_F1.get(m, (None, None))[1] for m in df.index]
    df.loc[our_name, "F1_oof"] = our_f1
    for col in ["params_M", "GFLOPs", "cpu_ms_b1"]:
        df[f"{col}_vs_ours"] = df[col] / df.loc[our_name, col]
    return df


# =============================================================================
# Roadmap extensions (E7 onward): sections 9-16
# =============================================================================
SAR_DIFF_LOCAL = [2, 3, 6, 7]       # SAR-difference channels inside the 8 SAR channels (raw 6,7,10,11)


# =============================================================================
# 9. Global normalisation statistics (train split only -> no leakage)
# =============================================================================
def _ids_key(df, indices):
    h = hashlib.md5((",".join(map(str, df["ID"])) + "|" + ",".join(indices)).encode()).hexdigest()
    return h[:10]


def compute_global_stats(train_df, data_dir, indices=(), clip=5.0):
    """Per-channel mean/std over the training images. Cached in results/."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"global_stats_{_ids_key(train_df, indices)}.json"
    if path.exists():
        d = json.load(open(path))
        return np.array(d["mean"], np.float32), np.array(d["std"], np.float32)
    s = s2 = n = 0.0
    for i in train_df["ID"]:
        raw = np.load(Path(data_dir) / f"{i}.npy").astype(np.float64)
        if indices:
            raw = np.concatenate([raw, compute_indices(raw, indices)], axis=-1)
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, raw.shape[-1])
        s = s + raw.sum(0)
        s2 = s2 + (raw ** 2).sum(0)
        n += raw.shape[0]
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    json.dump({"mean": mean.tolist(), "std": std.tolist()}, open(path, "w"))
    print(f"[global stats] {len(train_df)} images -> {path.name}")
    return mean.astype(np.float32), std.astype(np.float32)


# =============================================================================
# 10. Dataset with norm / SAR-selection options
# =============================================================================
class ExtDataset(Dataset):
    def __init__(self, df, data_dir, indices=(), norm="image", stats=None, sar="all", clip=5.0):
        self.df = df.reset_index(drop=True)
        self.dir = Path(data_dir)
        self.indices = list(indices)
        self.norm, self.sar, self.clip = norm, sar, clip
        self.stats = stats
        if norm == "global" and stats is None:
            raise ValueError("norm='global' needs stats=(mean, std)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        raw = np.load(self.dir / f"{row['ID']}.npy")
        if self.indices:
            raw = np.concatenate([raw, compute_indices(raw, self.indices)], axis=-1)
        if self.norm == "global":
            mean, std = self.stats
            img = np.nan_to_num(raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            img = np.clip((img - mean) / (std + 1e-6), -self.clip, self.clip)
        else:
            img = normalize_per_channel(raw)
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        s2 = torch.cat([img[0:4], img[12:]], dim=0)
        s1 = img[4:12]
        if self.sar == "diff":
            s1 = s1[SAR_DIFF_LOCAL]
        return s2, s1, torch.tensor(int(row["label"]), dtype=torch.float32)


def make_loaders_ext(data: Data, indices=(), aug=False, batch_size=32, norm="image",
                     sar="all", num_workers=0):
    stats = None
    if norm == "global":
        stats = compute_global_stats(data.train_split, data.data_dir, indices)
    tr = ExtDataset(data.train_split, data.data_dir, indices, norm, stats, sar)
    va = ExtDataset(data.val_split, data.data_dir, indices, norm, stats, sar)
    tr_ds = AugmentedDataset(tr) if aug else tr
    pin = torch.cuda.is_available()
    return (DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin),
            DataLoader(va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin))


# =============================================================================
# 11. Model (state-dict compatible with PretrainedMultimodalFlex for mnv3s)
# =============================================================================
def build_backbone(name, in_ch, pretrained):
    if name == "mnv3s":
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        feat = m.classifier[0].in_features
    elif name == "mnv3l":
        m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None)
        feat = m.classifier[0].in_features
    elif name == "effb0":
        m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        feat = m.classifier[1].in_features
    else:
        raise ValueError(name)
    m = adapt_first_conv(m, in_ch)
    m.classifier = nn.Identity()
    return m, feat


class ExtModel(nn.Module):
    def __init__(self, s2_ch=4, s1_ch=8, pretrained=True, backbone="mnv3s", img_size=None):
        super().__init__()
        self.img_size = img_size
        self.s2_backbone, feat = build_backbone(backbone, s2_ch, pretrained)
        self.s1_backbone, _ = build_backbone(backbone, s1_ch, pretrained)
        self.classifier = nn.Sequential(nn.Linear(2 * feat, 128), nn.Hardswish(),
                                        nn.Dropout(0.2), nn.Linear(128, 1))

    def _rs(self, x):
        if self.img_size and x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear",
                              align_corners=False)
        return x

    def forward(self, s2, s1):
        return self.classifier(torch.cat([self.s2_backbone(self._rs(s2)),
                                          self.s1_backbone(self._rs(s1))], dim=1))


def model_from_cfg(cfg, pretrained=None):
    s2c = 4 + len(cfg["indices"])
    s1c = 4 if cfg.get("sar", "all") == "diff" else 8
    return ExtModel(s2c, s1c, cfg["pretrained"] if pretrained is None else pretrained,
                    cfg.get("backbone", "mnv3s"), cfg.get("img_size"))


# =============================================================================
# 12. Losses
# =============================================================================
class BCESoftF1(nn.Module):
    """pos-weighted BCE + lam * (1 - soft F1 of the batch). lam=0 -> plain BCE."""

    def __init__(self, pos_weight, device, lam=0.5):
        super().__init__()
        self.bce = make_criterion(pos_weight, device)
        self.lam = lam

    def forward(self, logits, y, t_logits=None):
        loss = self.bce(logits, y)
        if self.lam > 0:
            p = torch.sigmoid(logits)
            tp = (p * y).sum()
            soft_f1 = 2 * tp / (p.sum() + y.sum() + 1e-6)
            loss = loss + self.lam * (1 - soft_f1)
        return loss


class DistillLoss(nn.Module):
    """alpha * hard loss + (1-alpha) * KD to teacher probabilities (binary, temperature T)."""

    def __init__(self, base, alpha=0.5, T=2.0):
        super().__init__()
        self.base, self.alpha, self.T = base, alpha, T

    def forward(self, logits, y, t_logits=None):
        hard = self.base(logits, y)
        if t_logits is None:
            return hard
        soft_t = torch.sigmoid(t_logits / self.T)
        kd = F.binary_cross_entropy_with_logits(logits / self.T, soft_t) * (self.T ** 2)
        return self.alpha * hard + (1 - self.alpha) * kd


# =============================================================================
# 13. Training core (cache + resume, optional teacher)
# =============================================================================
def _train_epoch(model, loader, criterion, optimizer, device, scheduler=None, teacher=None):
    model.train()
    tot, ys, ps = 0.0, [], []
    for s2, s1, y in loader:
        s2, s1, y = s2.to(device), s1.to(device), y.to(device)
        t_logits = None
        if teacher is not None:
            with torch.no_grad():
                t_logits = teacher(s2, s1).squeeze(1)
        optimizer.zero_grad()
        logits = model(s2, s1).squeeze(1)
        loss = criterion(logits, y, t_logits)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        tot += loss.item() * y.size(0)
        ps.append(torch.sigmoid(logits).detach().cpu().numpy())
        ys.append(y.cpu().numpy())
    y_, p_ = np.concatenate(ys), np.concatenate(ps)
    m = compute_metrics(y_, p_)
    return tot / len(loader.dataset), m["f1"], m["auc"]


def _cfg_dict(tag, seed, **kw):
    cfg = dict(tag=tag, seed=seed)
    cfg.update(kw)
    cfg["indices"] = list(cfg["indices"])
    return cfg


def _load_teacher(distill_from, cfg, device):
    meta = json.load(open(RESULTS_DIR / f"{run_name(distill_from, cfg['seed'])}.json"))
    tc = meta["config"]
    for k in ("norm", "sar", "indices"):
        if tc.get(k) != cfg[k]:
            raise ValueError(f"teacher/student differ in '{k}': {tc.get(k)} vs {cfg[k]}")
    t = model_from_cfg(tc, pretrained=False)
    t.load_state_dict(torch.load(CKPT_DIR / f"{run_name(distill_from, cfg['seed'])}_best.pth",
                                 map_location=device))
    t.to(device).eval()
    print(f"  teacher {distill_from}: {count_params(t)/1e6:.2f} M params")
    return t


def fit(data: Data, name, cfg, device=None, force=False):
    """Train/load one run on data.train_split / data.val_split. Returns history DataFrame.
    Also saves results/<name>_valprobs.npz with labels, probs at best epoch and at last epoch."""
    device = device or get_device()
    RESULTS_DIR.mkdir(exist_ok=True)
    CKPT_DIR.mkdir(exist_ok=True)
    hist_p, meta_p = RESULTS_DIR / f"{name}.csv", RESULTS_DIR / f"{name}.json"
    best_p, last_p = CKPT_DIR / f"{name}_best.pth", CKPT_DIR / f"{name}_last.pt"
    prob_p = RESULTS_DIR / f"{name}_valprobs.npz"

    if not force and meta_p.exists() and hist_p.exists():
        meta = json.load(open(meta_p))
        if meta.get("completed") and meta.get("config") == cfg:
            print(f"[cached] {name}: best val F1 = {meta['best_val_f1']:.4f} @ epoch {meta['best_epoch']}")
            return pd.read_csv(hist_p)
        print(f"[{name}] saved run differs or is incomplete -> re-running")

    set_seed(cfg["seed"])
    tr_loader, va_loader = make_loaders_ext(data, cfg["indices"], cfg["aug"], cfg["batch_size"],
                                            cfg["norm"], cfg["sar"])
    model = model_from_cfg(cfg).to(device)
    val_crit = make_criterion(data.pos_weight, device)
    base = BCESoftF1(data.pos_weight, device, lam=cfg["sf1_lam"] if cfg["loss"] == "bce_sf1" else 0.0)
    teacher = None
    if cfg["distill_from"]:
        teacher = _load_teacher(cfg["distill_from"], cfg, device)
        crit = DistillLoss(base, cfg["kd_alpha"], cfg["kd_T"])
    else:
        crit = base
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = None
    if cfg["sched"] == "cosine":
        steps = len(tr_loader)
        sched = make_scheduler(opt, cfg["epochs"] * steps, warmup_steps=steps)

    history, best_f1, best_ep, start = [], -1.0, 0, 1
    best_probs = None
    if last_p.exists() and not force:
        st = torch.load(last_p, map_location=device, weights_only=False)
        if st.get("config") == cfg:
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["optimizer"])
            if sched is not None and st.get("scheduler") is not None:
                sched.load_state_dict(st["scheduler"])
            history, best_f1, best_ep = st["history"], st["best_f1"], st["best_epoch"]
            best_probs = st.get("best_probs")
            start = st["epoch"] + 1
            _set_rng_state(st["rng"])
            print(f"[resume] {name}: from epoch {start}/{cfg['epochs']}")

    print(f"[{name}] params={count_params(model):,} | {cfg}")
    last_probs = labels = None
    for ep in range(start, cfg["epochs"] + 1):
        t0 = time.time()
        trl, trf, tra = _train_epoch(model, tr_loader, crit, opt, device, sched, teacher)
        m = evaluate(model, va_loader, val_crit, device, return_preds=True)
        labels, last_probs = m["labels"], m["probs"]
        history.append(dict(epoch=ep, train_loss=trl, train_f1=trf, train_auc=tra,
                            val_loss=m["loss"], val_accuracy=m["accuracy"],
                            val_precision=m["precision"], val_recall=m["recall"],
                            val_f1=m["f1"], val_auc=m["auc"],
                            lr=opt.param_groups[0]["lr"], time_sec=time.time() - t0))
        print(f"  [{name}] {ep:02d}/{cfg['epochs']} | train F1 {trf:.4f} | val F1 {m['f1']:.4f} "
              f"| AUC {m['auc']:.4f} | {history[-1]['time_sec']:.0f}s")
        if m["f1"] > best_f1:
            best_f1, best_ep, best_probs = m["f1"], ep, last_probs.copy()
            _atomic_torch_save(model.state_dict(), best_p)
        _atomic_write_csv(pd.DataFrame(history), hist_p)
        _atomic_torch_save({"config": cfg, "epoch": ep, "model": model.state_dict(),
                              "optimizer": opt.state_dict(),
                              "scheduler": sched.state_dict() if sched else None,
                              "history": history, "best_f1": best_f1, "best_epoch": best_ep,
                              "best_probs": best_probs, "rng": _get_rng_state()}, last_p)

    if labels is None:      # resumed after the final epoch finished
        m = evaluate(model, va_loader, val_crit, device, return_preds=True)
        labels, last_probs = m["labels"], m["probs"]
    np.savez(prob_p, labels=labels, probs_best=best_probs, probs_last=last_probs)
    _atomic_write_json({"completed": True, "config": cfg, "params": count_params(model),
                          "best_val_f1": best_f1, "best_epoch": best_ep,
                          "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}, meta_p)
    last_p.unlink(missing_ok=True)
    print(f"[{name}] done. best val F1 = {best_f1:.4f} @ epoch {best_ep}\n")
    return pd.DataFrame(history)


def run_experiment_ext(data: Data, tag, *, seed=SEED, indices=(), aug=True, sched="cosine",
                       lr=1e-3, weight_decay=1e-4, epochs=15, pretrained=True, batch_size=32,
                       norm="image", img_size=None, sar="all", loss="bce", backbone="mnv3s",
                       sf1_lam=0.5, distill_from=None, kd_alpha=0.5, kd_T=2.0,
                       force=False, device=None):
    cfg = _cfg_dict(tag, seed, indices=indices, aug=bool(aug), sched=sched, lr=lr,
                    weight_decay=weight_decay, epochs=epochs, pretrained=bool(pretrained),
                    batch_size=batch_size, norm=norm, img_size=img_size, sar=sar, loss=loss,
                    backbone=backbone, sf1_lam=sf1_lam, distill_from=distill_from,
                    kd_alpha=kd_alpha, kd_T=kd_T)
    return fit(data, run_name(tag, seed), cfg, device, force)


def run_grid_ext(data, configs: dict, seeds=(SEED,), **common):
    return {tag: [run_experiment_ext(data, tag, seed=s, **{**common, **c}) for s in seeds]
            for tag, c in configs.items()}


# =============================================================================
# 14. 5-fold CV with out-of-fold F1 (same protocol as the reference paper)
# =============================================================================
def run_cv(data: Data, tag, n_splits=5, seed=SEED, **kw):
    """Trains n_splits models on data.train_df. Returns (oof_df, per-fold summary).
    OOF probs use the LAST epoch (fixed protocol, no best-epoch selection on the fold)."""
    df = data.train_df.reset_index(drop=True)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=42)
    oof = np.full(len(df), np.nan)
    rows = []
    for k, (tr, va) in enumerate(skf.split(df, df["label"])):
        trd, vad = df.iloc[tr], df.iloc[va]
        d = Data(df, trd, vad, data.data_dir,
                 int((trd["label"] == 0).sum()) / int((trd["label"] == 1).sum()))
        h = run_experiment_ext(d, f"{tag}_cv{n_splits}f{k}", seed=seed, **kw)
        z = np.load(RESULTS_DIR / f"{run_name(f'{tag}_cv{n_splits}f{k}', seed)}_valprobs.npz")
        oof[va] = z["probs_last"]
        rows.append({"fold": k, "last_f1": h["val_f1"].iloc[-1], "best_f1": h["val_f1"].max()})
    y = df["label"].values
    ths = np.arange(0.2, 0.81, 0.01)
    f1s = [f1_score(y, oof >= t) for t in ths]
    summ = pd.DataFrame(rows).set_index("fold")
    print(f"\n[{tag}] OOF F1@0.5 = {f1_score(y, oof >= 0.5):.4f} | "
          f"OOF F1 @ tuned thr {ths[int(np.argmax(f1s))]:.2f} = {max(f1s):.4f}  "
          f"(tuned thr = paper protocol)")
    pd.DataFrame({"ID": df["ID"], "label": y, "oof_prob": oof}).to_csv(
        RESULTS_DIR / f"{tag}_cv{n_splits}_oof.csv", index=False)
    return pd.read_csv(RESULTS_DIR / f"{tag}_cv{n_splits}_oof.csv"), summ


# =============================================================================
# 15. Fusion (probability averaging of finished runs on the SAME val split) + TTA
# =============================================================================
def ensure_valprobs(data: Data, tag, seed=SEED, device=None):
    """Create results/<run>_valprobs.npz from the saved best checkpoint if it is missing
    (needed for runs made with the old runner, e.g. E6d). No training."""
    name = run_name(tag, seed)
    path = RESULTS_DIR / f"{name}_valprobs.npz"
    if path.exists():
        return path
    device = device or get_device()
    cfg = json.load(open(RESULTS_DIR / f"{name}.json"))["config"]
    model = model_from_cfg(cfg, pretrained=False)
    model.load_state_dict(torch.load(CKPT_DIR / f"{name}_best.pth", map_location=device))
    _, va = make_loaders_ext(data, cfg["indices"], False, cfg["batch_size"],
                             cfg.get("norm", "image"), cfg.get("sar", "all"))
    out = evaluate(model.to(device).eval(), va, make_criterion(data.pos_weight, device),
                   device, return_preds=True)
    np.savez(path, labels=out["labels"], probs_best=out["probs"])
    print(f"[valprobs] rebuilt for {name} (val F1 {out['f1']:.4f})")
    return path


def fuse_runs(tags, seed=SEED, which="probs_best", weights=None, data=None):
    """Average saved validation probabilities of several finished runs.
    Pass data=data to rebuild missing .npz files (old runs) from their best checkpoints."""
    if data is not None:
        for t in tags:
            ensure_valprobs(data, t, seed)
    zs = [np.load(RESULTS_DIR / f"{run_name(t, seed)}_valprobs.npz") for t in tags]
    y = zs[0]["labels"]
    for z in zs:
        assert np.array_equal(z["labels"], y), "runs were not evaluated on the same split/order"
    w = np.ones(len(zs)) / len(zs) if weights is None else np.array(weights) / np.sum(weights)
    p = sum(wi * z[which] for wi, z in zip(w, zs))
    rows = [{"run": t, "f1": compute_metrics(y, z[which])["f1"],
             "auc": compute_metrics(y, z[which])["auc"]} for t, z in zip(tags, zs)]
    m = compute_metrics(y, p)
    rows.append({"run": "FUSED(" + "+".join(tags) + ")", "f1": m["f1"], "auc": m["auc"]})
    return pd.DataFrame(rows).set_index("run"), y, p


@torch.no_grad()
def predict_tta(model, loader, device=None, n=8):
    """Average sigmoid over D4 transforms (n=4: rotations, n=8: + flips). Cost scales with n."""
    device = device or get_device()
    model.eval().to(device)
    ys, ps = [], []
    for s2, s1, y in loader:
        s2, s1 = s2.to(device), s1.to(device)
        acc = 0
        for k in range(4):
            for flip in ([False, True] if n == 8 else [False]):
                a, b = torch.rot90(s2, k, (2, 3)), torch.rot90(s1, k, (2, 3))
                if flip:
                    a, b = torch.flip(a, [3]), torch.flip(b, [3])
                acc = acc + torch.sigmoid(model(a, b).squeeze(1))
        ps.append((acc / n).cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def eval_run_tta(data: Data, tag, seed=SEED, n=8, device=None):
    cfg = json.load(open(RESULTS_DIR / f"{run_name(tag, seed)}.json"))["config"]
    device = device or get_device()
    model = model_from_cfg(cfg, pretrained=False)
    model.load_state_dict(torch.load(CKPT_DIR / f"{run_name(tag, seed)}_best.pth", map_location=device))
    _, va = make_loaders_ext(data, cfg["indices"], False, cfg["batch_size"],
                             cfg.get("norm", "image"), cfg.get("sar", "all"))
    y, p = predict_tta(model, va, device, n)
    return compute_metrics(y, p), y, p


# =============================================================================
# 16. Benchmarks for our variants + robust (repeat-min) latency
# =============================================================================
def _robust_cpu_ms(model, inputs, threads, repeats=5, runs=30):
    torch.set_num_threads(threads)
    meds = []
    for _ in range(repeats):
        med, _ = bench(copy.deepcopy(model), inputs, "cpu", warmup=10, runs=runs)
        meds.append(med)
    return float(np.min(meds)), float(np.median(meds))


def benchmark_variants(variants: dict, cpu_threads=4, repeats=5):
    """variants: {name: tag}  (finished runs; model rebuilt from its json config, random weights ok).
    Input stays 64x64; any upsampling happens inside the model, so FLOPs include it."""
    rows = []
    for name, tag in variants.items():
        cfg = json.load(open(RESULTS_DIR / f"{run_name(tag, SEED)}.json"))["config"]
        m = model_from_cfg(cfg, pretrained=False).eval()
        s2c, s1c = 4 + len(cfg["indices"]), (4 if cfg.get("sar", "all") == "diff" else 8)
        inp = (torch.randn(1, s2c, 64, 64), torch.randn(1, s1c, 64, 64))
        cmin, cmed = _robust_cpu_ms(m, inp, cpu_threads, repeats)
        rows.append({"model": name, "params_M": count_params(m) / 1e6, "size_MB": model_size_mb(m),
                     "GFLOPs": count_gflops(m, inp), "cpu_ms_min": cmin, "cpu_ms_median": cmed,
                     "threads": cpu_threads})
        gc.collect()
    return pd.DataFrame(rows).set_index("model")


def robust_reference_benchmark(cpu_threads=4, repeats=3, names=None):
    """Re-measure the reference models' CPU latency as min-of-medians (kills background-load spikes).
    Run plugged in, other apps closed. Use cpu_threads=1 for an edge-style number."""
    rows = []
    for name, (build, c_in) in REF_CONFIGS.items():
        if names and name not in names:
            continue
        m = build().eval()
        inp = (torch.randn(1, c_in, 256, 256),)
        cmin, cmed = _robust_cpu_ms(m, inp, cpu_threads, repeats, runs=20)
        rows.append({"model": name, "params_M": count_params(m) / 1e6,
                     "GFLOPs": count_gflops(m, inp), "cpu_ms_min": cmin, "cpu_ms_median": cmed,
                     "threads": cpu_threads})
        print("done:", name, f"{cmin:.0f} ms")
        del m
        gc.collect()
    return pd.DataFrame(rows).set_index("model")