from .lenet import LeNet5
from .googlenet import MiniGoogLeNet, InceptionBlock
from .resnet18 import ResNet18, BasicBlock
from .resnet50 import PretrainedResnet


def build_model(name, num_classes=21, img_size=64):
    name = name.lower()
    if name in ["lenet", "lenet_avg"]:
        return LeNet5(in_ch=3, img=img_size, num_classes=num_classes, pool="avg")
    elif name == "lenet_max":
        return LeNet5(in_ch=3, img=img_size, num_classes=num_classes, pool="max")
    elif name == "googlenet":
        return MiniGoogLeNet(in_ch=3, num_classes=num_classes)
    elif name == "resnet18":
        return ResNet18(num_classes=num_classes)
    elif name in ["resnet50_fe", "resnet50_feature_extract"]:
        return PretrainedResnet(num_classes=num_classes, freeze_backbone=True)
    elif name in ["resnet50_ft", "resnet50_fine_tune"]:
        return PretrainedResnet(num_classes=num_classes, freeze_backbone=False)
    else:
        raise ValueError(f"Khong tim thay model: {name}")


def n_params(module, trainable_only=False):
    ps = module.parameters()
    if trainable_only:
        ps = (p for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in ps)
