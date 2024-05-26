import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureDiscriminator(nn.Module):
    def __init__(self, ndim, use3d=False) -> None:
        super().__init__()
        
        self.ndim = ndim
        self.use3d = use3d

        if use3d:
            self.convs = nn.Sequential(
                nn.AdaptiveAvgPool3d((1, 1, 1))
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(ndim, ndim, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(ndim, ndim, 3, stride=2, padding=1),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(self.ndim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        x = self.convs(x)
        x = self.flatten(x)

        x = self.layers(x)

        return x