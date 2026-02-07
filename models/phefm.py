import torch
import torch.nn as nn

class HierarchicalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, stride=2, padding=1) # Downsample

    def forward(self, x):
        feat1 = torch.relu(self.enc1(x))
        feat2 = torch.relu(self.enc2(feat1))
        return feat1, feat2

class PHEFM(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_ir = HierarchicalEncoder()
        self.encoder_vis = HierarchicalEncoder()
        
        # Fusion layers for different hierarchies
        self.fuse_low = nn.Conv2d(64, 32, 1)
        self.fuse_high = nn.Conv2d(128, 64, 1)
        
        self.upsample = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.reconstruct = nn.Conv2d(96, 1, 3, padding=1)

    def forward(self, ir, vis):
        ir1, ir2 = self.encoder_ir(ir)
        vis1, vis2 = self.encoder_vis(vis)
        
        # Parallel Fusion
        f_low = self.fuse_low(torch.cat([ir1, vis1], dim=1))
        f_high = self.fuse_high(torch.cat([ir2, vis2], dim=1))
        
        # Merge hierarchies
        up_high = self.upsample(f_high)
        return self.reconstruct(torch.cat([f_low, up_high], dim=1))