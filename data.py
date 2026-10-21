import os
import zipfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def build_transforms(img_size, augment=True):
    """
    Tạo hai bộ transform:
      - train có data augmentation (RandomResizedCrop, HorizontalFlip, RandomRotation, ColorJitter) nếu augment=True
      - val và test luôn KHÔNG có data augmentation (chỉ resize cố định và chuẩn hoá)
    """
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def build_train_val_loaders(data_root, img_size, batch_size, augment=True,
                            val_ratio=0.1, num_workers=2, seed=42):
    """
    Tạo DataLoader cho tập train và validation từ thư mục VinaFood21.
    Nếu chưa có sẵn thư mục val/, tự động tách 10% từ tập train theo seed cố định.
    """
    root = Path(data_root)
    train_tf, eval_tf = build_transforms(img_size, augment=augment)

    if (root / "val").is_dir():
        train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
        val_ds   = datasets.ImageFolder(root / "val", transform=eval_tf)
        classes  = train_ds.classes
    else:
        full_train = datasets.ImageFolder(root / "train", transform=train_tf)
        full_eval  = datasets.ImageFolder(root / "train", transform=eval_tf)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(full_train), generator=g).tolist()
        n_val = int(len(idx) * val_ratio)
        train_ds = Subset(full_train, idx[n_val:])
        val_ds   = Subset(full_eval, idx[:n_val])
        classes  = full_train.classes

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return train_dl, val_dl, classes


def build_test_loader(data_root, img_size, batch_size, num_workers=2):
    """
    Tạo DataLoader cho tập test.
    """
    root = Path(data_root)
    _, eval_tf = build_transforms(img_size, augment=False)
    test_ds = datasets.ImageFolder(root / "test", transform=eval_tf)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return test_dl, test_ds.classes


def download_vinafood21(target_dir="VinaFood21"):
    """
    Tự động tải bộ dữ liệu VinaFood21 từ Google Drive nếu chưa có trên máy / Colab.
    """
    target_path = Path(target_dir)
    if (target_path / "train").is_dir():
        print(f"Đã tìm thấy dữ liệu tại: {target_path.resolve()}")
        return target_path

    print("Dữ liệu chưa tồn tại, đang tải VinaFood21 từ Google Drive...")
    import gdown
    url = "https://drive.google.com/uc?id=1UpZOf0XlwvB4rKpyZ35iwTA8oWHqDBbR"
    zip_path = "VinaFood21.zip"
    gdown.download(url, zip_path, quiet=False)

    print("Đang giải nén VinaFood21.zip...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")
    print("Giải nén thành công!")
    return target_path
