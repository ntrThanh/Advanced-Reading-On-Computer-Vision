import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EfficientNet(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_b0(weights=weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if not name.startswith("classifier.1"):
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def unfreeze_last_blocks(self, num_blocks=1):
        self.freeze_backbone()
        total_blocks = len(self.model.features)
        for idx in range(max(0, total_blocks - num_blocks), total_blocks):
            for param in self.model.features[idx].parameters():
                param.requires_grad = True

    def get_param_groups(self, lr_backbone=1e-4, lr_classifier=1e-3):
        backbone_params = []
        classifier_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("classifier.1"):
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": classifier_params, "lr": lr_classifier}
        ]
