import torch
import torch.nn as nn
from model.module.point_branch import Pointbranch
import numpy as np

class PVTblock(nn.Module):
    def __init__(self, in_channels, out_channels, voxel_resolution, window_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.voxel_resolution = voxel_resolution
        self.window_size = window_size        
        # TODO: voxel_branch
        self.voxel_branch = None
        self.point_branch = Pointbranch(in_channels, out_channels)
        
    def forward(self, x):
        fused_features = self.point_branch(x) #+ self.voxel_branch
        
        return fused_features
    

