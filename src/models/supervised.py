import torch.nn as nn
from torchvision.models import resnet18


class SupervisedResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
