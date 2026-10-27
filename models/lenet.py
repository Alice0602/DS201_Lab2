import torch
import torch.nn as nn


class LeNet5(nn.Module):
    # LeNet-5 cho anh mau (in_ch=3, 64x64, 21 classes)
    def __init__(self, in_ch=3, img=64, num_classes=21, pool="avg"):
        super().__init__()
        Pool = nn.AvgPool2d if pool == "avg" else nn.MaxPool2d
        self.c1 = nn.Conv2d(in_ch, 6, kernel_size=5)
        self.s2 = Pool(2, 2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5)
        self.s4 = Pool(2, 2)

        self.n_flat = self._flat_dim(in_ch, img)
        self.f5 = nn.Linear(self.n_flat, 120)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)

    def _flat_dim(self, in_ch, img):
        with torch.no_grad():
            z = torch.zeros(1, in_ch, img, img)
            z = self.s4(self.c3(self.s2(self.c1(z))))
        return int(z.flatten(1).shape[1])

    def forward(self, x):
        x = self.s2(torch.relu(self.c1(x)))
        x = self.s4(torch.relu(self.c3(x)))
        x = x.flatten(1)
        x = torch.relu(self.f5(x))
        x = torch.relu(self.f6(x))
        return self.out(x)


if __name__ == "__main__":
    m = LeNet5(in_ch=3, img=64, num_classes=21, pool="avg")
    x = torch.rand(2, 3, 64, 64)
    out = m(x)
    total_params = sum(p.numel() for p in m.parameters())
    print(f"LeNet-5: {tuple(x.shape)} -> {tuple(out.shape)} | params: {total_params:,}")
    assert total_params == 339_421
