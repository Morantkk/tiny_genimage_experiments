import torch
import torch.nn as nn
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


class SpatialFrequencyResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.rgb_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.freq_branch = FrequencyBranch()
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
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

    def forward(self, x):
        rgb_feat = torch.flatten(self.rgb_backbone(x), 1)
        freq_feat = self.freq_branch(self.compute_freq_map(x))
        return self.classifier(torch.cat([rgb_feat, freq_feat], dim=1))


def build_model(num_classes: int):
    return SpatialFrequencyResNet18(num_classes)


def main():
    parser = default_argparser("resnet18_frequency")
    args = parser.parse_args()
    run_training(args, "resnet18_frequency", build_model)


if __name__ == "__main__":
    main()
