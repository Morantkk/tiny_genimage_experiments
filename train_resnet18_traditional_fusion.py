import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import models, transforms

from common import (
    GENERATOR_NAMES,
    cleanup_distributed,
    compute_metrics,
    default_argparser,
    gather_arrays,
    is_dist,
    prepare_image_cache,
    save_eval_outputs,
    set_seed,
    setup_distributed,
)


FEATURE_GROUPS = {
    "first20": [
        "gray_mean",
        "gray_var",
        "laplacian_var",
        "fft_mean_energy",
        *[f"gray_hist_{i:02d}" for i in range(16)],
    ],
    "color": [
        *[f"rgb_mean_{c}" for c in ["r", "g", "b"]],
        *[f"rgb_std_{c}" for c in ["r", "g", "b"]],
        *[f"rgb_skew_{c}" for c in ["r", "g", "b"]],
        *[f"hsv_hist_{i:02d}" for i in range(12)],
    ],
    "texture": [
        *[f"lbp_hist_{i:02d}" for i in range(16)],
        "sobel_mean",
        "sobel_std",
        "sobel_p90",
        "laplacian_abs_mean",
    ],
    "frequency": [
        *[f"fft_radial_{i:02d}" for i in range(8)],
        "fft_low_energy",
        "fft_mid_energy",
        "fft_high_energy",
        "fft_high_low_ratio",
    ],
}
FEATURE_GROUPS["all"] = (
    FEATURE_GROUPS["first20"]
    + FEATURE_GROUPS["color"]
    + FEATURE_GROUPS["texture"]
    + FEATURE_GROUPS["frequency"]
)
ALL_FEATURE_NAMES = FEATURE_GROUPS["all"]


def rgb_to_hsv_np(rgb):
    rgb = rgb.astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    delta = maxc - minc
    h = np.zeros_like(maxc)
    mask = delta > 1e-6
    r_mask = mask & (maxc == r)
    g_mask = mask & (maxc == g)
    b_mask = mask & (maxc == b)
    h[r_mask] = ((g[r_mask] - b[r_mask]) / delta[r_mask]) % 6
    h[g_mask] = ((b[g_mask] - r[g_mask]) / delta[g_mask]) + 2
    h[b_mask] = ((r[b_mask] - g[b_mask]) / delta[b_mask]) + 4
    h = h / 6.0
    s = np.where(maxc > 1e-6, delta / (maxc + 1e-6), 0.0)
    v = maxc
    return np.stack([h, s, v], axis=-1)


def extract_all_features(image_path):
    image = Image.open(image_path).convert("RGB")
    # Handcrafted statistics use a fixed analysis size for speed and comparability.
    image = image.resize((256, 256), Image.BILINEAR)
    rgb = np.asarray(image).astype(np.float32)
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)

    gray_norm = gray / 255.0
    gray_hist, _ = np.histogram(gray, bins=16, range=(0, 255), density=False)
    gray_hist = gray_hist.astype(np.float32) / max(gray_hist.sum(), 1.0)

    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    gy, gx = np.gradient(gray)
    sobel = np.sqrt(gx * gx + gy * gy)

    fft_mag = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray_norm))))
    h, w = fft_mag.shape
    yy, xx = np.indices((h, w))
    rr = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    rr = rr / (rr.max() + 1e-6)
    radial = []
    for i in range(8):
        mask = (rr >= i / 8.0) & (rr < (i + 1) / 8.0)
        radial.append(float(fft_mag[mask].mean()) if np.any(mask) else 0.0)
    low = float(fft_mag[rr < 0.20].mean())
    mid = float(fft_mag[(rr >= 0.20) & (rr < 0.55)].mean())
    high = float(fft_mag[rr >= 0.55].mean())

    rgb_scaled = rgb / 255.0
    rgb_mean = rgb_scaled.mean(axis=(0, 1))
    rgb_std = rgb_scaled.std(axis=(0, 1))
    rgb_skew = (((rgb_scaled - rgb_mean) / (rgb_std + 1e-6)) ** 3).mean(axis=(0, 1))
    hsv = rgb_to_hsv_np(rgb)
    hsv_hists = []
    for channel in range(3):
        hist, _ = np.histogram(hsv[..., channel], bins=4, range=(0.0, 1.0), density=False)
        hsv_hists.extend((hist.astype(np.float32) / max(hist.sum(), 1.0)).tolist())

    center = gray[1:-1, 1:-1]
    codes = np.zeros_like(center, dtype=np.uint8)
    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]
    for bit, nb in enumerate(neighbors):
        codes |= ((nb >= center).astype(np.uint8) << bit)
    lbp_hist, _ = np.histogram(codes, bins=16, range=(0, 256), density=False)
    lbp_hist = lbp_hist.astype(np.float32) / max(lbp_hist.sum(), 1.0)

    values = {
        "gray_mean": float(gray_norm.mean()),
        "gray_var": float(gray_norm.var()),
        "laplacian_var": float((lap / 255.0).var()),
        "fft_mean_energy": float(fft_mag.mean()),
        "fft_low_energy": low,
        "fft_mid_energy": mid,
        "fft_high_energy": high,
        "fft_high_low_ratio": high / (low + 1e-6),
        "sobel_mean": float((sobel / 255.0).mean()),
        "sobel_std": float((sobel / 255.0).std()),
        "sobel_p90": float(np.percentile(sobel / 255.0, 90)),
        "laplacian_abs_mean": float(np.abs(lap / 255.0).mean()),
    }
    for i, v in enumerate(gray_hist):
        values[f"gray_hist_{i:02d}"] = float(v)
    for i, v in enumerate(radial):
        values[f"fft_radial_{i:02d}"] = float(v)
    for prefix, arr in [("rgb_mean", rgb_mean), ("rgb_std", rgb_std), ("rgb_skew", rgb_skew)]:
        for c, v in zip(["r", "g", "b"], arr):
            values[f"{prefix}_{c}"] = float(v)
    for i, v in enumerate(hsv_hists):
        values[f"hsv_hist_{i:02d}"] = float(v)
    for i, v in enumerate(lbp_hist):
        values[f"lbp_hist_{i:02d}"] = float(v)
    return values


def read_manifest(path):
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_feature_matrix(rows, feature_names):
    features = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    for idx, row in enumerate(rows):
        values = extract_all_features(row["image_path"])
        features[idx] = np.array([values[name] for name in feature_names], dtype=np.float32)
        if (idx + 1) % 1000 == 0:
            print(f"extracted handcrafted features {idx + 1}/{len(rows)}", flush=True)
    return features


def prepare_raw_all_feature_cache(cache_root):
    out_dir = cache_root / "traditional_features"
    train_raw_npz = out_dir / "train_all_raw.npz"
    val_raw_npz = out_dir / "validation_all_raw.npz"
    meta_json = out_dir / "meta_all_raw.json"
    if train_raw_npz.exists() and val_raw_npz.exists() and meta_json.exists():
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_manifest(cache_root / "manifests" / "train.csv")
    val_rows = read_manifest(cache_root / "manifests" / "validation.csv")
    train_raw = build_feature_matrix(train_rows, ALL_FEATURE_NAMES)
    val_raw = build_feature_matrix(val_rows, ALL_FEATURE_NAMES)
    np.savez_compressed(train_raw_npz, features=train_raw.astype(np.float32))
    np.savez_compressed(val_raw_npz, features=val_raw.astype(np.float32))
    meta_json.write_text(
        json.dumps({"feature_names": ALL_FEATURE_NAMES}, indent=2),
        encoding="utf-8",
    )


def prepare_feature_cache(cache_root, feature_set, rank):
    feature_names = FEATURE_GROUPS[feature_set]
    out_dir = cache_root / "traditional_features"
    train_npz = out_dir / f"train_{feature_set}.npz"
    val_npz = out_dir / f"validation_{feature_set}.npz"
    scaler_json = out_dir / f"scaler_{feature_set}.json"
    meta_json = out_dir / f"meta_{feature_set}.json"
    if train_npz.exists() and val_npz.exists() and scaler_json.exists():
        return

    if is_dist() and rank != 0:
        dist.barrier()
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_raw_all_feature_cache(cache_root)
    train_all = np.load(out_dir / "train_all_raw.npz")["features"].astype(np.float32)
    val_all = np.load(out_dir / "validation_all_raw.npz")["features"].astype(np.float32)
    indices = [ALL_FEATURE_NAMES.index(name) for name in feature_names]
    train_raw = train_all[:, indices]
    val_raw = val_all[:, indices]
    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    train_scaled = (train_raw - mean) / std
    val_scaled = (val_raw - mean) / std
    np.savez_compressed(train_npz, features=train_scaled.astype(np.float32))
    np.savez_compressed(val_npz, features=val_scaled.astype(np.float32))
    scaler_json.write_text(
        json.dumps({"mean": mean.tolist(), "std": std.tolist()}, indent=2),
        encoding="utf-8",
    )
    meta_json.write_text(
        json.dumps({"feature_set": feature_set, "feature_names": feature_names}, indent=2),
        encoding="utf-8",
    )
    if is_dist():
        dist.barrier()


def prepare_requested_feature_caches(data_root, cache_root, feature_sets):
    prepare_image_cache(data_root, cache_root, rank=0)
    prepare_raw_all_feature_cache(cache_root)
    for feature_set in feature_sets:
        print(f"prepare feature cache: {feature_set}", flush=True)
        prepare_feature_cache(cache_root, feature_set, rank=0)


class ManifestImageFeatureDataset(Dataset):
    def __init__(self, manifest_path, feature_path, image_size, train):
        self.rows = read_manifest(manifest_path)
        self.features = np.load(feature_path)["features"].astype(np.float32)
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(int(image_size * 1.14)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        hand = torch.from_numpy(self.features[idx])
        return self.transform(image), hand, int(row["label"])


def build_feature_loaders(args, rank, world_size):
    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    prepare_image_cache(data_root, cache_root, rank)
    prepare_feature_cache(cache_root, args.feature_set, rank)

    feat_dir = cache_root / "traditional_features"
    train_set = ManifestImageFeatureDataset(
        cache_root / "manifests" / "train.csv",
        feat_dir / f"train_{args.feature_set}.npz",
        args.image_size,
        True,
    )
    val_set = ManifestImageFeatureDataset(
        cache_root / "manifests" / "validation.csv",
        feat_dir / f"validation_{args.feature_set}.npz",
        args.image_size,
        False,
    )
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True) if is_dist() else None
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False) if is_dist() else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_sampler


class ResNetTraditionalFusion(nn.Module):
    def __init__(self, num_classes, hand_dim, fusion="concat", proj_dim=512):
        super().__init__()
        self.fusion = fusion
        if fusion != "hand_only":
            resnet = models.resnet18(weights=None)
            self.rgb_backbone = nn.Sequential(*list(resnet.children())[:-1])
            self.rgb_norm = nn.LayerNorm(512)
        self.hand_proj = nn.Sequential(
            nn.Linear(hand_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, proj_dim),
            nn.LayerNorm(proj_dim),
        )
        if fusion == "concat":
            self.classifier = nn.Sequential(
                nn.LayerNorm(512 + proj_dim),
                nn.Linear(512 + proj_dim, 512),
                nn.GELU(),
                nn.Dropout(0.30),
                nn.Linear(512, num_classes),
            )
        elif fusion == "gated":
            self.gate = nn.Sequential(nn.Linear(512 + proj_dim, 512), nn.Sigmoid())
            self.classifier = nn.Sequential(
                nn.LayerNorm(512),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.30),
                nn.Linear(256, num_classes),
            )
        elif fusion == "film":
            self.modulator = nn.Linear(proj_dim, 1024)
            self.classifier = nn.Sequential(
                nn.LayerNorm(512),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Dropout(0.30),
                nn.Linear(256, num_classes),
            )
        elif fusion == "hand_only":
            self.hand_only = nn.Sequential(
                nn.Linear(proj_dim, 256),
                nn.GELU(),
                nn.Dropout(0.25),
                nn.Linear(256, num_classes),
            )
        else:
            raise ValueError(f"Unsupported fusion mode: {fusion}")

    def forward(self, image, hand):
        hand_feat = self.hand_proj(hand)
        if self.fusion == "hand_only":
            return self.hand_only(hand_feat)
        rgb_feat = torch.flatten(self.rgb_backbone(image), 1)
        rgb_feat = self.rgb_norm(rgb_feat)
        if self.fusion == "concat":
            return self.classifier(torch.cat([rgb_feat, hand_feat], dim=1))
        if self.fusion == "gated":
            gate = self.gate(torch.cat([rgb_feat, hand_feat], dim=1))
            fused = gate * rgb_feat + (1.0 - gate) * hand_feat
            return self.classifier(fused)
        gamma, beta = self.modulator(hand_feat).chunk(2, dim=1)
        fused = rgb_feat * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)
        return self.classifier(fused)


@torch.no_grad()
def evaluate_feature_model(model, loader, device, use_amp):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_loss, total_seen = 0.0, 0
    criterion = nn.CrossEntropyLoss()
    for images, hand, labels in loader:
        images = images.to(device, non_blocking=True)
        hand = hand.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(images, hand)
            loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_seen += images.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
    all_labels = gather_arrays(all_labels)
    all_preds = gather_arrays(all_preds)
    all_probs = gather_arrays(all_probs)
    return total_loss / max(total_seen, 1), np.array(all_labels), np.array(all_preds), np.array(all_probs)


def run_feature_training(args, experiment_name):
    local_rank, rank, world_size = setup_distributed()
    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    train_loader, val_loader, train_sampler = build_feature_loaders(args, rank, world_size)
    hand_dim = len(FEATURE_GROUPS[args.feature_set])
    model = ResNetTraditionalFusion(
        num_classes=len(GENERATOR_NAMES),
        hand_dim=hand_dim,
        fusion=args.fusion,
        proj_dim=args.proj_dim,
    ).to(device)
    if is_dist():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    out_dir = Path(args.output_root) / experiment_name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        config = vars(args).copy()
        config["feature_dim"] = hand_dim
        config["class_names"] = GENERATOR_NAMES
        (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss, seen = 0.0, 0
        for images, hand, labels in train_loader:
            images = images.to(device, non_blocking=True)
            hand = hand.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(images, hand)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            seen += images.size(0)
        scheduler.step()

        val_loss, labels, preds, probs = evaluate_feature_model(model, val_loader, device, use_amp)
        if rank == 0:
            metrics = compute_metrics(labels, preds, probs)
            row = {
                "epoch": epoch,
                "train_loss": running_loss / max(seen, 1),
                "val_loss": val_loss,
                **metrics,
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if metrics["multiclass_macro_f1"] > best_f1:
                best_f1 = metrics["multiclass_macro_f1"]
                ckpt_model = model.module if hasattr(model, "module") else model
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": ckpt_model.state_dict(),
                        "metrics": metrics,
                        "class_names": GENERATOR_NAMES,
                        "feature_set": args.feature_set,
                        "fusion": args.fusion,
                        "feature_dim": hand_dim,
                    },
                    out_dir / "best_model.pth",
                )
                save_eval_outputs(out_dir / "best_eval", labels, preds, probs, metrics)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    cleanup_distributed()


def main():
    parser = default_argparser("resnet18_traditional_fusion")
    parser.add_argument("--feature-set", choices=sorted(FEATURE_GROUPS.keys()), default="first20")
    parser.add_argument("--fusion", choices=["concat", "gated", "film", "hand_only"], default="concat")
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare_requested_feature_caches(
            Path(args.data_root),
            Path(args.cache_root),
            [args.feature_set],
        )
        return
    experiment_name = f"resnet18_traditional_{args.feature_set}_{args.fusion}"
    run_feature_training(args, experiment_name)


if __name__ == "__main__":
    main()
