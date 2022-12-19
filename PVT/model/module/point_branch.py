import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RelativeAttention(nn.Module):
    def __init__(self, latent_vec_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.latent_vec_dim = latent_vec_dim
        self.head_dim = int(latent_vec_dim / num_heads)
        self.query = nn.Conv1d(latent_vec_dim, latent_vec_dim, 1, bias=False)
        self.key = nn.Conv1d(latent_vec_dim, latent_vec_dim, 1, bias=False)
        self.value = nn.Conv1d(latent_vec_dim, latent_vec_dim, 1, bias=False)
        self.mlp = nn.Conv1d(latent_vec_dim, latent_vec_dim, 1, bias=False)
        self.bn = nn.BatchNorm1d(latent_vec_dim)
        self.rel_pos = nn.Linear(int(latent_vec_dim/num_heads), latent_vec_dim)
        self.scale = torch.sqrt(self.head_dim * torch.ones(1))
                
    def forward(self, x, rel_pos):
        # x.shape = (B, C, N)
        batch_size = x.size(0)
        N = x.shape[2]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3) # (B, H, N, C)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1) # (B, H, C, N)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,3,1) # (B, H, C, N)
        rel_pos = torch.cat([rel_pos, rel_pos, rel_pos, rel_pos], dim = 1).to(q.device)
        rel_pos = rel_pos.view(batch_size, self.num_heads, N, N)
        # rel_pos = rel_pos.view(batch_size, -1, self.num_heads, self.head_dim).permute(0,2,1,3)
        # rel_pos = self.rel_pos(rel_pos)
        
        attention = torch.softmax((q @ k)/self.scale.to(q.device), dim = -1) # (B, N, N)
        x_r = v @ attention # (B, C, N)
        x_r = x_r.permute(0,3,1,2).reshape(batch_size, self.latent_vec_dim, -1)
        x_r = torch.relu(self.bn(self.mlp(x_r)))
        
        return x_r
        
class Pointbranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.ra1 = RelativeAttention(out_channels)
        
        self.conv_fuse = nn.Sequential(nn.Conv1d(out_channels*2, out_channels, kernel_size=1, bias=False),
                                       nn.BatchNorm1d(out_channels),
                                       nn.LeakyReLU(negative_slope=0.2)
                                       )
        
    def forward(self, x):
        # x.shape = (B, C, N)
        pos = x[:,:3,:]
        pos = pos.permute(0,2,1)
        rel_pos = pos[:,:,None,:] - pos[:,None,:,:]
        rel_pos = rel_pos.sum(dim=-1)
        x = torch.relu(self.bn1(self.conv1(x)))
        ra_x = self.ra1(x, rel_pos)
        x = x + ra_x
        x = torch.cat((x, ra_x), dim=1)
        x = self.conv_fuse(x)
        return x
    
if __name__ == '__main__':
    pb = Pointbranch(6, 768)
    x = torch.ones(32, 6, 768)
    pos = torch.rand(32, 4, 1, 256)
    print(pb(x).shape)
    
