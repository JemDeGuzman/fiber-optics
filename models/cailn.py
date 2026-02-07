import torch
import torch.nn as nn

class CrossAttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Conv2d(dim, dim, kernel_size=1)
        self.key = nn.Conv2d(dim, dim, kernel_size=1)
        self.value = nn.Conv2d(dim, dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        # x1: Query source, x2: Key/Value source
        b, c, h, w = x1.shape
        q = self.query(x1).view(b, c, -1) 
        k = self.key(x2).view(b, c, -1)   
        v = self.value(x2).view(b, c, -1) 

        # Attention weights
        attn = self.softmax(torch.bmm(q.transpose(1, 2), k))
        out = torch.bmm(v, attn.transpose(1, 2))
        return out.view(b, c, h, w)

class CAILN(nn.Module):
    def __init__(self):
        super().__init__()
        self.extract = nn.Sequential(nn.Conv2d(1, 64, 3, padding=1), nn.ReLU())
        self.cross_attn = CrossAttentionModule(64)
        self.merge = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, ir, vis):
        f_ir = self.extract(ir)
        f_vis = self.extract(vis)
        
        # IR attends to Vis
        interacted = self.cross_attn(f_ir, f_vis)
        
        # Concatenate and reconstruct
        combined = torch.cat([interacted, f_vis], dim=1)
        return self.merge(combined)