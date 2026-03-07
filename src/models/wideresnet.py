import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.drop_rate = drop_rate
        self.equal_in_out = (in_planes == out_planes)
        self.shortcut = None
        if not self.equal_in_out:
            self.shortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
            )

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut = x if self.equal_in_out else self.shortcut(x)

        out = self.conv1(out)
        out = self.relu2(self.bn2(out))

        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        out = self.conv2(out)
        return shortcut + out


class NetworkBlock(nn.Module):
    def __init__(self, num_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                block(
                    in_planes if i == 0 else out_planes,
                    out_planes,
                    stride if i == 0 else 1,
                    drop_rate
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=2, num_classes=10, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor

        channels = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, channels[0], channels[1], BasicBlock, 1, drop_rate)
        self.block2 = NetworkBlock(n, channels[1], channels[2], BasicBlock, 2, drop_rate)
        self.block3 = NetworkBlock(n, channels[2], channels[3], BasicBlock, 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(channels[3])
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        self.out_channels = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.out_channels)
        return self.fc(out)


def get_model(num_classes=10):
    return WideResNet(depth=28, widen_factor=2, num_classes=num_classes, drop_rate=0.0)