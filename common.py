import argparse
import csv
import json
import os
import random
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


GENERATOR_NAMES = [
    "Real",
    "ADM",
    "BigGAN",
    "GLIDE",
    "Midjourney",
    "SD14",
    "SD15",
    "VQDM",
    "Wukong",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_dist() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    if not is_dist():
        return 0, 0, 1
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def rank0_print(rank: int, text: str) -> None:
    if rank == 0:
        print(text, flush=True)


def prepare_image_cache(data_root: Path, cache_root: Path, rank: int = 0) -> None:
    manifest_dir = cache_root / "manifests"
    train_manifest = manifest_dir / "train.csv"
    val_manifest = manifest_dir / "validation.csv"
    if train_manifest.exists() and val_manifest.exists():
        return

    if is_dist() and rank != 0:
        dist.barrier()
        return

    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    data_dir = data_root / "data"

    for split, pattern in [("train", "train-*.parquet"), ("validation", "validation-*.parquet")]:
        rows = []
        out_split = cache_root / "images" / split
        files = sorted(data_dir.glob(pattern))
        for shard_idx, parquet_file in enumerate(files):
            table = pq.read_table(parquet_file, columns=["image", "generator"])
            images = table["image"].to_pylist()
            labels = table["generator"].to_pylist()
            for row_idx, (image_obj, label) in enumerate(zip(images, labels)):
                label_name = GENERATOR_NAMES[int(label)]
                out_dir = out_split / label_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{parquet_file.stem}_{row_idx:05d}.jpg"
                if not out_file.exists():
                    img = Image.open(BytesIO(image_obj["bytes"])).convert("RGB")
                    img.save(out_file, quality=95)
                rows.append(
                    {
                        "image_path": str(out_file),
                        "label": int(label),
                        "label_name": label_name,
                        "binary_label": 0 if int(label) == 0 else 1,
                        "split": split,
                    }
                )
            print(f"cached {split} shard {shard_idx + 1}/{len(files)}: {parquet_file.name}", flush=True)

        manifest_path = manifest_dir / f"{split}.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["image_path", "label", "label_name", "binary_label", "split"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {manifest_path} rows={len(rows)}", flush=True)

    if is_dist():
        dist.barrier()


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_path: Path, image_size: int, train: bool):
        self.rows = []
        with manifest_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append(row)

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
        return self.transform(image), int(row["label"])


def build_loaders(args, rank: int, world_size: int):
    data_root = Path(args.data_root)
    cache_root = Path(args.cache_root)
    prepare_image_cache(data_root, cache_root, rank)

    train_set = ManifestImageDataset(cache_root / "manifests" / "train.csv", args.image_size, True)
    val_set = ManifestImageDataset(cache_root / "manifests" / "validation.csv", args.image_size, False)
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


def gather_arrays(values):
    if not is_dist():
        return values
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, values)
    merged = []
    for part in gathered:
        merged.extend(part)
    return merged


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_loss, total_seen = 0.0, 0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(images)
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
    avg_loss = total_loss / max(total_seen, 1)
    return avg_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels, preds, probs):
    multi_precision, multi_recall, multi_f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(labels, preds, average="weighted")
    metrics = {
        "multiclass_accuracy": float(accuracy_score(labels, preds)),
        "multiclass_macro_precision": float(multi_precision),
        "multiclass_macro_recall": float(multi_recall),
        "multiclass_macro_f1": float(multi_f1),
        "multiclass_weighted_f1": float(weighted_f1),
    }

    y_bin = (labels != 0).astype(int)
    p_bin = (preds != 0).astype(int)
    b_precision, b_recall, b_f1, _ = precision_recall_fscore_support(
        y_bin, p_bin, average="binary", zero_division=0
    )
    metrics.update(
        {
            "binary_accuracy": float(accuracy_score(y_bin, p_bin)),
            "binary_precision_fake": float(b_precision),
            "binary_recall_fake": float(b_recall),
            "binary_f1_fake": float(b_f1),
            "real_recall": float(((p_bin == 0) & (y_bin == 0)).sum() / max((y_bin == 0).sum(), 1)),
        }
    )
    try:
        metrics["binary_auc_fake"] = float(roc_auc_score(y_bin, 1.0 - probs[:, 0]))
    except Exception:
        metrics["binary_auc_fake"] = None
    try:
        metrics["multiclass_ovr_auc"] = float(
            roc_auc_score(labels, probs, multi_class="ovr", average="macro")
        )
    except Exception:
        metrics["multiclass_ovr_auc"] = None
    return metrics


def save_eval_outputs(out_dir: Path, labels, preds, probs, metrics: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = confusion_matrix(labels, preds, labels=list(range(len(GENERATOR_NAMES))))
    np.savetxt(out_dir / "confusion_matrix_multiclass.csv", cm, fmt="%d", delimiter=",")
    binary_cm = confusion_matrix((labels != 0).astype(int), (preds != 0).astype(int), labels=[0, 1])
    np.savetxt(out_dir / "confusion_matrix_binary.csv", binary_cm, fmt="%d", delimiter=",")

    report = classification_report(
        labels,
        preds,
        labels=list(range(len(GENERATOR_NAMES))),
        target_names=GENERATOR_NAMES,
        zero_division=0,
        output_dict=True,
    )
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "pred", "label_name", "pred_name", "binary_label", "binary_pred"])
        for y, p in zip(labels.tolist(), preds.tolist()):
            writer.writerow([y, p, GENERATOR_NAMES[y], GENERATOR_NAMES[p], int(y != 0), int(p != 0)])


def default_argparser(experiment_name: str):
    parser = argparse.ArgumentParser(experiment_name)
    parser.add_argument("--data-root", default="/data2/liweixiang/chenfakun/ai_detected/dataset/Tiny-GenImage")
    parser.add_argument("--cache-root", default="/data2/liweixiang/chenfakun/ai_detected/dataset/Tiny-GenImage_cache")
    parser.add_argument("--output-root", default="/data2/liweixiang/chenfakun/ai_detected/code/tiny_genimage_experiments/outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true")
    return parser


def run_training(args, experiment_name: str, build_model):
    local_rank, rank, world_size = setup_distributed()
    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    train_loader, val_loader, train_sampler = build_loaders(args, rank, world_size)
    model = build_model(num_classes=len(GENERATOR_NAMES)).to(device)
    if is_dist():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)

    out_dir = Path(args.output_root) / experiment_name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss, seen = 0.0, 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            seen += images.size(0)
        scheduler.step()

        val_loss, labels, preds, probs = evaluate(model, val_loader, device, use_amp)
        if rank == 0:
            metrics = compute_metrics(labels, preds, probs)
            train_loss = running_loss / max(seen, 1)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
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
                    },
                    out_dir / "best_model.pth",
                )
                save_eval_outputs(out_dir / "best_eval", labels, preds, probs, metrics)

            (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    cleanup_distributed()
