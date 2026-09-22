import torch
import torch.nn as nn
import timm


class MobileNetV4_GRU(nn.Module):
    def __init__(
        self,
        num_classes=50,
        model_name="mobilenetv4_conv_small",
        pretrained=True,
        hidden_size=256,
        num_layers=1,
        bidirectional=True,
        dropout=0.3
    ):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        with torch.no_grad():
            dummy = torch.zeros(2, 3, 112, 112)
            feature_dim = self.backbone(dummy).shape[1]

        self.early_fusion = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        gru_out_dim = hidden_size * 2 if bidirectional else hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def freeze_backbone(self, freeze=True):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def forward(self, x):
        if x.dim() == 5:
            if x.size(1) == 3 and x.size(2) != 3:
                x = x.permute(0, 2, 1, 3, 4)

        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)

        features = self.backbone(x)
        features = features.reshape(b, t, -1)

        features = features.permute(0, 2, 1)
        fused = self.early_fusion(features)
        fused = fused.permute(0, 2, 1)

        gru_out, _ = self.gru(fused)
        out = self.classifier(gru_out.mean(dim=1))
        return out


if __name__ == "__main__":
    model = MobileNetV4_GRU(num_classes=50, pretrained=False)
    dummy_input = torch.randn(2, 3, 16, 112, 112)
    output = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
