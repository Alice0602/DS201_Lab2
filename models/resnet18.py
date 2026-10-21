import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Residual block cơ bản của ResNet-18:
    y = F(x) + shortcut(x): cộng trước, ReLU sau.
    """
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Shortcut identity khi khớp shape, projection 1x1 conv khi đổi channel hoặc stride > 1
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return torch.relu(out)


class ResNet18(nn.Module):
    """
    ResNet-18 đầy đủ [2, 2, 2, 2] cho 21 lớp món ăn VinaFood21.
    Tổng tham số chuẩn xác: 11,187,285.
    """
    def __init__(self, num_classes=21):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.layer1 = self._stage(64, 64, stride=1)
        self.layer2 = self._stage(64, 128, stride=2)
        self.layer3 = self._stage(128, 256, stride=2)
        self.layer4 = self._stage(256, 512, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _stage(in_ch, out_ch, stride):
        return nn.Sequential(
            BasicBlock(in_ch, out_ch, stride),
            BasicBlock(out_ch, out_ch, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


if __name__ == "__main__":
    m = ResNet18(num_classes=21)
    x = torch.rand(2, 3, 96, 96)
    out = m(x)
    total_params = sum(p.numel() for p in m.parameters())
    print(f"ResNet-18: input {tuple(x.shape)} -> logits {tuple(out.shape)}")
    print(f"Tổng tham số: {total_params:,}")
    assert total_params == 11_187_285, "Số tham số ResNet-18 phải là 11,187,285"
