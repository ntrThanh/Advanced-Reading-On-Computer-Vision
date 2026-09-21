import torch
import torch.nn as nn


class ParallelBranch(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels=16, kernel_size=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(0.001),
            nn.MaxPool3d(2),
            nn.Conv3d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(0.001),
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        return self.block(x)


class SimpleCNN3D_v2(nn.Module):
    def __init__(self, num_classes=50, in_channels=3, k1=1, k2=3, k3=5):
        super().__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.001),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.001)
        )

        self.parallel1 = ParallelBranch(32, 16, 16, kernel_size=k1)
        self.parallel2 = ParallelBranch(32, 8, 16, kernel_size=k2)
        self.parallel3 = ParallelBranch(32, 4, 16, kernel_size=k3)

        self.pool_final = nn.MaxPool3d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 2, 2))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 2 * 2 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        p1 = self.parallel1(x)
        p2 = self.parallel2(x)
        p3 = self.parallel3(x)
        x = p1 + p2 + p3
        x = self.pool_final(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = SimpleCNN3D_v2(num_classes=50)
    dummy_input = torch.randn(2, 3, 16, 112, 112)
    out = model(dummy_input)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", out.shape)
