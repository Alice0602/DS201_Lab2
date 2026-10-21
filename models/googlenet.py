import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """
    Inception block với 4 nhánh song song:
      - Nhánh 1: Conv 1x1
      - Nhánh 2: Conv 1x1 (bottleneck) -> Conv 3x3
      - Nhánh 3: Conv 1x1 (bottleneck) -> Conv 5x5
      - Nhánh 4: MaxPool 3x3 -> Conv 1x1
    Đầu ra nối các nhánh theo chiều channel (dim=1).
    """
    def __init__(self, in_ch, f1, f3r, f3, f5r, f5, fp):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, f1, 1), nn.ReLU()
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, f3r, 1), nn.ReLU(),
            nn.Conv2d(f3r, f3, 3, padding=1), nn.ReLU()
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, f5r, 1), nn.ReLU(),
            nn.Conv2d(f5r, f5, 5, padding=2), nn.ReLU()
        )
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, fp, 1), nn.ReLU()
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class MiniGoogLeNet(nn.Module):
    """
    GoogLeNet rút gọn với 3 Inception blocks và Global Average Pooling (GAP) cho ảnh 64x64.
    """
    def __init__(self, in_ch=3, num_classes=21):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        )
        # Block 1: in_ch=64 -> out_ch = 32 + 48 + 16 + 16 = 112
        self.incept1 = InceptionBlock(64, f1=32, f3r=32, f3=48, f5r=8, f5=16, fp=16)
        
        # Block 2: in_ch=112 -> out_ch = 64 + 96 + 32 + 32 = 224
        self.incept2 = InceptionBlock(112, f1=64, f3r=64, f3=96, f5r=16, f5=32, fp=32)
        self.pool = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # Block 3: in_ch=224 -> out_ch = 128 + 160 + 32 + 32 = 352
        self.incept3 = InceptionBlock(224, f1=128, f3r=96, f3=160, f5r=16, f5=32, fp=32)
        
        # Head: GAP + Linear
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(352, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.incept1(x)
        x = self.incept2(x)
        x = self.pool(x)
        x = self.incept3(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)


if __name__ == "__main__":
    m = MiniGoogLeNet(in_ch=3, num_classes=21)
    x = torch.rand(2, 3, 64, 64)
    out = m(x)
    total_params = sum(p.numel() for p in m.parameters())
    print(f"GoogLeNet: input {tuple(x.shape)} -> logits {tuple(out.shape)}")
    print(f"Tổng tham số: {total_params:,}")
