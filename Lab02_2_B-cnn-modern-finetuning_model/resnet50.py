import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ResNet50(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.model = resnet50(weights=weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def unfreeze_last_blocks(self, num_blocks=1):
        self.freeze_backbone()
        blocks = ["layer4", "layer3", "layer2", "layer1"][:num_blocks]
        for name, param in self.model.named_parameters():
            if any(name.startswith(b) for b in blocks):
                param.requires_grad = True

    def get_param_groups(self, lr_backbone=1e-4, lr_classifier=1e-3):
        backbone_params = []
        classifier_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("fc"):
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": classifier_params, "lr": lr_classifier}
        ]
