import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) # Downsample to 1/2
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) # Downsample to 1/4
        )

    def forward(self, x):
        return self.enc2(self.enc1(x))

class FiberClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Three separate encoders for Luster, Roughness, and Tensile
        self.enc_luster = HierarchicalEncoder(1)
        self.enc_rough = HierarchicalEncoder(1)
        self.enc_tensile = HierarchicalEncoder(1)
        
        # Fusion layer: Combines 64*3 features
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1) # Global Average Pooling to 1x1
        )
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2) # Output: [Abaca Score, Daratex Score]
        )

    def forward(self, luster, rough, tensile):
        l_feat = self.enc_luster(luster)
        r_feat = self.enc_rough(rough)
        t_feat = self.enc_tensile(tensile)
        
        # Concatenate along channel dimension
        fused = torch.cat([l_feat, r_feat, t_feat], dim=1)
        
        bottleneck = self.fusion(fused)
        flat = torch.flatten(bottleneck, 1)
        
        return self.classifier(flat)