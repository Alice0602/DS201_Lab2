import os
import zipfile
from pathlib import Path
from PIL import Image, ImageFile

# Cho phep Pillow doc anh bi cat ngan/thieu byte (tranh OSError truncated)
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def safe_pil_loader(path):
    try:
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGBA").convert("RGB")
    except OSError:
        img = Image.open(path)
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

    # Neu o Colab va da co san /content/VinaFood21 voi 21 lop
    colab_parent = Path("/content/VinaFood21")
    if colab_parent.resolve() != target_path.resolve() and (colab_parent / "train").is_dir():
        classes_found = [d for d in (colab_parent / "train").iterdir() if d.is_dir()]
        if len(classes_found) == 21:
            if not target_path.exists():
                try:
                    os.symlink(str(colab_parent), str(target_path))
                    print(f"Lien ket du lieu tu {colab_parent} (21 lop)")
                    return
                except Exception:
                    pass

    # Kiem tra thu muc hien tai da co du 21 lop chua
    if (target_path / "train").is_dir():
        classes_found = [d for d in (target_path / "train").iterdir() if d.is_dir()]
        if len(classes_found) == 21:
            return
        else:
            print(f"Phat hien thu muc {target_path} chi co {len(classes_found)}/21 lop, giai nen lai...")
            import shutil
            shutil.rmtree(target_path, ignore_errors=True)

    # Tim file zip
    zip_candidates = [Path("VinaFood21.zip"), Path("/content/VinaFood21.zip")]
    zip_path = None
    for zc in zip_candidates:
        if zc.is_file() and zc.stat().st_size > 100_000_000:
            zip_path = zc
            break

    if zip_path is None:
        import gdown
        url = "https://drive.google.com/uc?id=1UpZOf0XlwvB4rKpyZ35iwTA8oWHqDBbR"
        print("Downloading VinaFood21...")
        zip_path = Path("VinaFood21.zip")
        gdown.download(url, str(zip_path), quiet=False)

    print(f"Extracting {zip_path}...")
    res = os.system(f"unzip -q -o {zip_path} -d .")
    if res != 0:
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(".")
    print("Done!")
