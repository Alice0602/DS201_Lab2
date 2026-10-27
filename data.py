import os
import zipfile
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def safe_pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGBA").convert("RGB")


def build_transforms(img_size, augment=True):
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
    root = Path(data_root)
    train_tf, eval_tf = build_transforms(img_size, augment=augment)

    if (root / "val").is_dir():
        train_ds = datasets.ImageFolder(root / "train", transform=train_tf, loader=safe_pil_loader)
        val_ds   = datasets.ImageFolder(root / "val", transform=eval_tf, loader=safe_pil_loader)
        classes  = train_ds.classes
    else:
        full_train = datasets.ImageFolder(root / "train", transform=train_tf, loader=safe_pil_loader)
        full_eval  = datasets.ImageFolder(root / "train", transform=eval_tf, loader=safe_pil_loader)
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
    root = Path(data_root)
    _, eval_tf = build_transforms(img_size, augment=False)
    test_ds = datasets.ImageFolder(root / "test", transform=eval_tf, loader=safe_pil_loader)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return test_dl, test_ds.classes


def download_vinafood21(target_dir="VinaFood21"):
    target_path = Path(target_dir)
    if (target_path / "train").is_dir():
        return

    import gdown
    zip_name = "VinaFood21.zip"
    if not os.path.exists(zip_name):
        url = "https://drive.google.com/uc?id=1UpZOf0XlwvB4rKpyZ35iwTA8oWHqDBbR"
        print("Downloading VinaFood21...")
        gdown.download(url, zip_name, quiet=False)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_name, "r") as z:
        z.extractall(".")
    print("Done!")
