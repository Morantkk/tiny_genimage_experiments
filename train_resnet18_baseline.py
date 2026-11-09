import torch.nn as nn
from torchvision import models

from common import default_argparser, run_training


class ResNet18Baseline(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


def build_model(num_classes: int):
    return ResNet18Baseline(num_classes)


def main():
    parser = default_argparser("resnet18_baseline")
    args = parser.parse_args()
    run_training(args, "resnet18_baseline", build_model)


if __name__ == "__main__":
    main()
