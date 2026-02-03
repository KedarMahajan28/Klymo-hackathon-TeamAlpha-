import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import os
import sys

# Add SwinIR to path
sys.path.append("SwinIR")
from models.network_swinir import SwinIR


class TemporalFusionModule(nn.Module):
    def __init__(self, in_channels=27, out_channels=3):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 24, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, out_channels, 3, 1, 1)
        )

    def forward(self, x):
        return self.fusion(x)


class MemoryEfficientSwinIR(nn.Module):
    def __init__(self, in_channels=27, pretrained_path=None, use_checkpointing=False):
        super().__init__()

        self.temporal_fusion = TemporalFusionModule(in_channels, 3)

        self.swinir = SwinIR(
            upscale=4,
            in_chans=3,
            img_size=48,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6],
            embed_dim=120,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv"
        )

        if pretrained_path and os.path.exists(pretrained_path):
            ckpt = torch.load(pretrained_path, map_location="cpu")
            if "params" in ckpt:
                self.swinir.load_state_dict(ckpt["params"], strict=False)

    def forward(self, x):
        x = self.temporal_fusion(x)

        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        return self.swinir(x)
