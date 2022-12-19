import torch
import torch.nn as nn
from torch.nn import functional as F
from model.module.model_mlp import ModelMLP
from model.PVTblock import PVTblock

def create_pvt_blocks(blocks, in_channels, eps=0, 
                               width_multiplier=1, voxel_resolution_multiplier=1, model=''):
    r, vr = width_multiplier, voxel_resolution_multiplier
    layers, concat_channels = [], 0
    
    for out_channels, num_blocks, voxel_resolution in blocks:
        out_channels = r * out_channels
        for _ in range(num_blocks):
            if voxel_resolution is None:
                layers.append(ModelMLP(in_channels, out_channels))
            elif model == 'PVTblock':
                layers.append(PVTblock(in_channels, out_channels, voxel_resolution= vr * voxel_resolution, window_size=3))
            in_channels = out_channels
            concat_channels += out_channels

    return layers, in_channels, concat_channels

class pvt(nn.Module):
    blocks_list = [[64,1,30], [128,2,15], [512,1,None], [1024,1,None]] # out_channels, num_blocks, voxel_resolution
    
    def __init__(self, num_classes=40, width_multiplier=1, voxel_resolution_multiplier=1):
        super().__init__()
        self.in_channels = 6
        layers, in_channels, concat_channels = create_pvt_blocks(
                                                        blocks=self.blocks_list, in_channels=self.in_channels,
                                                        width_multiplier=width_multiplier, voxel_resolution_multiplier=voxel_resolution_multiplier,
                                                        model='PVTblock'
        )
        
        self.pvt_features = nn.ModuleList(layers)
        
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(concat_channels + in_channels *2, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024)
        )
        
        self.linear1 = nn.Linear(1024, 512)
        self.dp1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.dp2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        
    def forward(self, x):
        num_points, batch_size = x.size(2), x.size(0)
        coords = x[:,:3,:]
        out_features_list = []
        for i in range(len(self.pvt_features)):
            x = self.pvt_features[i](x)
            out_features_list.append(x)
        
        out_features_list.append(x.max(dim=-1, keepdim=True).values.repeat([1,1,num_points]))
        out_features_list.append(x.mean(dim=-1, keepdim=True).view(batch_size, -1).unsqueeze(-1).repeat([1,1,num_points]))
        
        features = torch.cat(out_features_list, dim=1)
        features = F.leaky_relu(self.conv_fuse(features))
        features = F.adaptive_max_pool1d(features, 1).view(batch_size, -1)
        features = F.leaky_relu(self.bn1(self.linear1(features)))
        features = self.dp1(features)
        features = F.leaky_relu(self.bn2(self.linear2(features)))
        features = self.dp2(features)
        features = self.linear3(features)
        return features
    
if __name__ == '__main__':
    pvt = pvt()
    x = torch.ones(32, 6, 768)
    print(pvt(x).shape)