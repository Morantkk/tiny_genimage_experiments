import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from common import default_argparser, run_training


class FrequencyBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return torch.flatten(self.net(x), 1)


class FusionResNetFrequencyPatchTransformer(nn.Module):
    def __init__(self, num_classes: int, patch_grid: int = 2):
        super().__init__()
        self.patch_grid = patch_grid

        global_resnet = models.resnet18(weights=None)
        self.global_backbone = nn.Sequential(*list(global_resnet.children())[:-1])

        patch_resnet = models.resnet18(weights=None)
        self.patch_backbone = nn.Sequential(*list(patch_resnet.children())[:-1])
        self.pos_embed = nn.Parameter(torch.zeros(1, patch_grid * patch_grid, 512))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.freq_branch = FrequencyBranch()

        self.classifier = nn.Sequential(
            nn.LayerNorm(512 + 512 + 128),
            nn.Linear(512 + 512 + 128, 512),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(512, num_classes),
        )

    def compute_freq_map(self, x):
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        fft = torch.fft.fft2(gray)
        fft = torch.fft.fftshift(fft)
        mag = torch.log1p(torch.abs(fft))
        _, _, h, w = mag.shape
        mask = torch.ones_like(mag)
        h0, h1 = int(h * 0.4), int(h * 0.6)
        w0, w1 = int(w * 0.4), int(w * 0.6)
        mask[:, :, h0:h1, w0:w1] = 0
        mag = mag * mask
        mean = mag.mean(dim=(2, 3), keepdim=True)
        std = mag.std(dim=(2, 3), keepdim=True)
        return (mag - mean) / (std + 1e-6)

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

        global_feat = torch.flatten(self.global_backbone(x), 1)
        freq_feat = self.freq_branch(self.compute_freq_map(x))

        patches = self.make_patches(x)
        patch_feat = torch.flatten(self.patch_backbone(patches), 1)
        patch_feat = patch_feat.view(self.patch_grid * self.patch_grid, b, -1).permute(1, 0, 2)
        patch_feat = self.transformer(patch_feat + self.pos_embed).mean(dim=1)

        fused = torch.cat([global_feat, patch_feat, freq_feat], dim=1)
        return self.classifier(fused)


def build_model(num_classes: int):
    return FusionResNetFrequencyPatchTransformer(num_classes)


def main():
    parser = default_argparser("fusion_model")
    parser.add_argument("--patch-grid", type=int, default=2)
    args = parser.parse_args()

    def factory(num_classes: int):
        return FusionResNetFrequencyPatchTransformer(num_classes, patch_grid=args.patch_grid)

    run_training(args, "fusion_model", factory)


if __name__ == "__main__":
    main()
