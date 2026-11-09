import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import models

from common import (
    GENERATOR_NAMES,
    cleanup_distributed,
    compute_metrics,
    default_argparser,
    gather_arrays,
    is_dist,
    save_eval_outputs,
    set_seed,
    setup_distributed,
)
from train_resnet18_traditional_fusion import (
    FEATURE_GROUPS,
    build_feature_loaders,
    evaluate_feature_model,
)


class OptimizedPatchTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        patch_grid=2,
        transformer_layers=2,
        transformer_heads=8,
        feature_dim=512,
        use_cls=True,
        use_global=True,
        dropout=0.25,
    ):
        super().__init__()
        self.patch_grid = patch_grid
        self.use_cls = use_cls
        self.use_global = use_global
        patch_resnet = models.resnet18(weights=None)
        self.patch_backbone = nn.Sequential(*list(patch_resnet.children())[:-1])
        if use_global:
            global_resnet = models.resnet18(weights=None)
            self.global_backbone = nn.Sequential(*list(global_resnet.children())[:-1])
        token_count = patch_grid * patch_grid + int(use_cls) + int(use_global)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim)) if use_cls else None
        self.pos_embed = nn.Parameter(torch.zeros(1, token_count, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=transformer_heads,
            dim_feedforward=feature_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def make_patches(self, x):
        b, c, h, w = x.shape
        ph, pw = h // self.patch_grid, w // self.patch_grid
        patches = []
        for i in range(self.patch_grid):
            for j in range(self.patch_grid):
                patch = x[:, :, i * ph : (i + 1) * ph, j * pw : (j + 1) * pw]
                patch = F.interpolate(patch, size=(h, w), mode="bilinear", align_corners=False)
                patches.append(patch)
        return torch.cat(patches, dim=0)

    def encode(self, x):
        b = x.size(0)
        patches = self.make_patches(x)
        patch_feat = torch.flatten(self.patch_backbone(patches), 1)
        tokens = patch_feat.view(self.patch_grid * self.patch_grid, b, -1).permute(1, 0, 2)
        token_list = []
        if self.use_cls:
            token_list.append(self.cls_token.expand(b, -1, -1))
        token_list.append(tokens)
        if self.use_global:
            global_feat = torch.flatten(self.global_backbone(x), 1).unsqueeze(1)
            token_list.append(global_feat)
        tokens = torch.cat(token_list, dim=1)
        tokens = self.transformer(tokens + self.pos_embed[:, : tokens.size(1)])
        if self.use_cls:
            return tokens[:, 0]
        return tokens.mean(dim=1)

    def forward(self, x):
        return self.classifier(self.encode(x))


class PatchTraditionalFusion(nn.Module):
    def __init__(
        self,
        num_classes,
        hand_dim,
        fusion="concat",
        patch_grid=3,
        transformer_layers=2,
        transformer_heads=8,
        feature_dim=512,
        use_cls=True,
        use_global=True,
    ):
        super().__init__()
        self.fusion = fusion
        self.patch_encoder = OptimizedPatchTransformer(
            num_classes=num_classes,
            patch_grid=patch_grid,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            feature_dim=feature_dim,
            use_cls=use_cls,
            use_global=use_global,
        )
        self.patch_encoder.classifier = nn.Identity()
        self.hand_proj = nn.Sequential(
            nn.Linear(hand_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        if fusion == "concat":
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim * 2),
                nn.Linear(feature_dim * 2, 512),
                nn.GELU(),
                nn.Dropout(0.30),
                nn.Linear(512, num_classes),
            )
        elif fusion == "gated":
            self.gate = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.Sigmoid())
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, 256),
                nn.GELU(),
                nn.Dropout(0.30),
                nn.Linear(256, num_classes),
            )
        elif fusion == "film":
            self.modulator = nn.Linear(feature_dim, feature_dim * 2)
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, 256),
                nn.GELU(),
                nn.Dropout(0.30),
                nn.Linear(256, num_classes),
            )
        elif fusion == "crossattn":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=feature_dim, num_heads=transformer_heads, dropout=0.1, batch_first=True
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, 256),
                nn.GELU(),
                nn.Dropout(0.30),
                nn.Linear(256, num_classes),
            )
        else:
            raise ValueError(f"Unsupported fusion: {fusion}")

    def forward(self, image, hand):
        patch_feat = self.patch_encoder.encode(image)
        hand_feat = self.hand_proj(hand)
        if self.fusion == "concat":
            return self.classifier(torch.cat([patch_feat, hand_feat], dim=1))
        if self.fusion == "gated":
            gate = self.gate(torch.cat([patch_feat, hand_feat], dim=1))
            return self.classifier(gate * patch_feat + (1.0 - gate) * hand_feat)
        if self.fusion == "film":
            gamma, beta = self.modulator(hand_feat).chunk(2, dim=1)
            fused = patch_feat * (1.0 + 0.1 * torch.tanh(gamma)) + 0.1 * torch.tanh(beta)
            return self.classifier(fused)
        query = patch_feat.unsqueeze(1)
        key_value = hand_feat.unsqueeze(1)
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        return self.classifier(patch_feat + attn_out.squeeze(1))


@torch.no_grad()
def evaluate_image_model(model, loader, device, use_amp):
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
    return total_loss / max(total_seen, 1), np.array(all_labels), np.array(all_preds), np.array(all_probs)


def save_checkpoint(model, out_dir, epoch, metrics, args):
    ckpt_model = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": ckpt_model.state_dict(),
            "metrics": metrics,
            "class_names": GENERATOR_NAMES,
            "args": vars(args),
        },
        out_dir / "best_model.pth",
    )


def run_patch_training(args, experiment_name):
    local_rank, rank, world_size = setup_distributed()
    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    from common import build_loaders

    train_loader, val_loader, train_sampler = build_loaders(args, rank, world_size)
    model = OptimizedPatchTransformer(
        num_classes=len(GENERATOR_NAMES),
        patch_grid=args.patch_grid,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        use_cls=args.use_cls,
        use_global=args.use_global,
        dropout=args.dropout,
    ).to(device)
    if is_dist():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    run_loop(args, experiment_name, model, train_loader, val_loader, train_sampler, device, use_amp, rank, image_only=True)


def run_patch_feature_training(args, experiment_name):
    local_rank, rank, world_size = setup_distributed()
    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    train_loader, val_loader, train_sampler = build_feature_loaders(args, rank, world_size)
    hand_dim = len(FEATURE_GROUPS[args.feature_set])
    model = PatchTraditionalFusion(
        num_classes=len(GENERATOR_NAMES),
        hand_dim=hand_dim,
        fusion=args.fusion,
        patch_grid=args.patch_grid,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        use_cls=args.use_cls,
        use_global=args.use_global,
    ).to(device)
    if is_dist():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    run_loop(args, experiment_name, model, train_loader, val_loader, train_sampler, device, use_amp, rank, image_only=False)


def run_loop(args, experiment_name, model, train_loader, val_loader, train_sampler, device, use_amp, rank, image_only):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=use_amp)
    out_dir = Path(args.output_root) / experiment_name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        config = vars(args).copy()
        config["class_names"] = GENERATOR_NAMES
        (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best_f1 = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        running_loss, seen = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if image_only:
                images, labels = batch
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(enabled=use_amp):
                    logits = model(images)
                    loss = criterion(logits, labels)
                batch_size = images.size(0)
            else:
                images, hand, labels = batch
                images = images.to(device, non_blocking=True)
                hand = hand.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with autocast(enabled=use_amp):
                    logits = model(images, hand)
                    loss = criterion(logits, labels)
                batch_size = images.size(0)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * batch_size
            seen += batch_size
        scheduler.step()
        if image_only:
            val_loss, labels, preds, probs = evaluate_image_model(model, val_loader, device, use_amp)
        else:
            val_loss, labels, preds, probs = evaluate_feature_model(model, val_loader, device, use_amp)
        if rank == 0:
            metrics = compute_metrics(labels, preds, probs)
            row = {"epoch": epoch, "train_loss": running_loss / max(seen, 1), "val_loss": val_loss, **metrics}
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if metrics["multiclass_macro_f1"] > best_f1:
                best_f1 = metrics["multiclass_macro_f1"]
                save_checkpoint(model, out_dir, epoch, metrics, args)
                save_eval_outputs(out_dir / "best_eval", labels, preds, probs, metrics)
            (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    cleanup_distributed()


def main():
    parser = default_argparser("patch_transformer_optimized")
    parser.add_argument("--mode", choices=["patch", "fusion"], default="patch")
    parser.add_argument("--patch-grid", type=int, default=3)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--use-cls", action="store_true")
    parser.add_argument("--no-global", dest="use_global", action="store_false")
    parser.set_defaults(use_global=True)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--feature-set", choices=sorted(FEATURE_GROUPS.keys()), default="frequency")
    parser.add_argument("--fusion", choices=["concat", "gated", "film", "crossattn"], default="concat")
    parser.add_argument("--experiment-name", default=None)
    args = parser.parse_args()
    if args.experiment_name:
        experiment_name = args.experiment_name
    elif args.mode == "patch":
        cls_name = "cls" if args.use_cls else "mean"
        global_name = "global" if args.use_global else "noglobal"
        experiment_name = f"patch_opt_g{args.patch_grid}_l{args.transformer_layers}_{cls_name}_{global_name}"
    else:
        experiment_name = f"patch_opt_freq_{args.fusion}"
    if args.mode == "patch":
        run_patch_training(args, experiment_name)
    else:
        run_patch_feature_training(args, experiment_name)


if __name__ == "__main__":
    main()
