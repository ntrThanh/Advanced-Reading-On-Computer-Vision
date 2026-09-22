import torch
import torch.nn as nn


class SimpleCNN3D_Sequential(nn.Module):
    def __init__(self, num_classes=50, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.001),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.001),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.001),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.001),
            nn.MaxPool3d(2),
            nn.AdaptiveAvgPool3d((2, 2, 2))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 2 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        if x.dim() == 5:
            if x.size(1) != 3 and x.size(2) == 3:
                x = x.permute(0, 2, 1, 3, 4)
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = SimpleCNN3D_Sequential(num_classes=50)
    dummy_input = torch.randn(2, 3, 16, 112, 112)
    out = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", out.shape)
