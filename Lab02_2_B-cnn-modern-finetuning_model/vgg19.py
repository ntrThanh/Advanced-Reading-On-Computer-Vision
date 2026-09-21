import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class VGG19(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=False):
        super().__init__()
        weights = VGG19_Weights.DEFAULT if pretrained else None
        self.model = vgg19(weights=weights)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if not name.startswith("classifier.6"):
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def unfreeze_last_blocks(self, num_layers=4):
        self.freeze_backbone()
        total_features = len(self.model.features)
        for idx in range(max(0, total_features - num_layers), total_features):
            for param in self.model.features[idx].parameters():
                param.requires_grad = True

    def get_param_groups(self, lr_backbone=1e-4, lr_classifier=1e-3):
        backbone_params = []
        classifier_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("classifier.6"):
                classifier_params.append(param)
            else:
                backbone_params.append(param)
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": classifier_params, "lr": lr_classifier}
        ]
