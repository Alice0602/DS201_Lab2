import torch
import torch.nn as nn


class PretrainedResnet(nn.Module):
    """
    Sử dụng backbone ResNet-50 pretrained từ HuggingFace (microsoft/resnet-50).
    Thay classifier gốc bằng nn.Linear(2048, num_classes).
    Hỗ trợ 2 chế độ:
      - Feature Extraction: freeze_backbone=True (chỉ học classifier, 43,029 tham số)
      - Fine-tuning: freeze_backbone=False (học toàn bộ 23.5 triệu tham số)
    """
    def __init__(self, num_classes=21, freeze_backbone=False):
        super().__init__()
        from transformers import ResNetForImageClassification
        base = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.resnet = base.resnet
        self.classifier = nn.Linear(2048, num_classes, bias=True)

        if freeze_backbone:
            for p in self.resnet.parameters():
                p.requires_grad = False

    def forward(self, images):
        features = self.resnet(images).pooler_output  # [B, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1)   # [B, 2048]
        return self.classifier(features)              # [B, num_classes]


if __name__ == "__main__":
    try:
        fe = PretrainedResnet(num_classes=21, freeze_backbone=True)
        ft = PretrainedResnet(num_classes=21, freeze_backbone=False)
        total_fe = sum(p.numel() for p in fe.parameters())
        trainable_fe = sum(p.numel() for p in fe.parameters() if p.requires_grad)
        trainable_ft = sum(p.numel() for p in ft.parameters() if p.requires_grad)
        print(f"ResNet-50 FE: Tổng {total_fe:,} | Cần cập nhật: {trainable_fe:,}")
        print(f"ResNet-50 FT: Cần cập nhật: {trainable_ft:,}")
        assert total_fe == 23_551_061
        assert trainable_fe == 43_029
        assert trainable_ft == 23_551_061
    except Exception as e:
        print("Cần cài đặt transformers để chạy kiểm tra ResNet-50:", e)
