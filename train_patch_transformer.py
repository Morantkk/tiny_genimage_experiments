import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from common import default_argparser, run_training


class PatchTransformerResNet18(nn.Module):
    def __init__(
        self,
        num_classes: int,
        patch_grid: int = 2,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        feature_dim: int = 512,
    ):
        super().__init__()
        self.patch_grid = patch_grid
        resnet = models.resnet18(weights=None)
        self.patch_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_grid * patch_grid, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=transformer_heads,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
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

    def forward(self, x):
        b = x.size(0)
        patches = self.make_patches(x)
        feats = torch.flatten(self.patch_backbone(patches), 1)
        feats = feats.view(self.patch_grid * self.patch_grid, b, -1).permute(1, 0, 2)
        feats = feats + self.pos_embed
        feats = self.transformer(feats)
        pooled = feats.mean(dim=1)
        return self.classifier(pooled)


def build_model(num_classes: int):
    return PatchTransformerResNet18(num_classes)


def main():
    parser = default_argparser("patch_transformer")
    parser.add_argument("--patch-grid", type=int, default=2)
    args = parser.parse_args()

    def factory(num_classes: int):
        return PatchTransformerResNet18(num_classes, patch_grid=args.patch_grid)

    run_training(args, "patch_transformer", factory)


if __name__ == "__main__":
    main()
