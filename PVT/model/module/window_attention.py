from typing import Union

import torch
from torch import nn


class WindowAttention(nn.Module):
    def __init__(self, 
                 num_channels: int,
                 patch_size: int,
                 use_cyclic_shift: bool,
                 num_heads: int, 
                 dropout: float=0, 
                 bias: bool=True, 
                 shift: bool=True,
                 add_bias_kv: bool=False, 
                 add_zero_attn: bool=False, 
                 vdim: Union[int, None]=None,
                 kdim: Union[int, None]=None): 
        super(WindowAttention, self).__init__()
        
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.use_cyclic_shift = use_cyclic_shift
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.shift = shift
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.vdim = vdim
        self.kdim = kdim
        
        self.attn = nn.MultiheadAttention(self.num_channels * (self.patch_size ** 3), 
                                          num_heads, dropout, bias, 
                                          add_bias_kv, add_zero_attn, vdim, kdim)
        
    def forward(self, x):
        if x.ndim == 4:
            squeeze_later = True
            x = x.unsqueeze(0)
        elif x.ndim == 5:
            squeeze_later = False
        else:
            raise ValueError(f'Expected a tensor of (B, C, Rx, Ry, Rz) or (C, Rx, Ry, Rz) but got {x.shape} instead.')
        
        batch_size, _, *resolution = x.shape
        resolution = torch.tensor(resolution)
        
        num_patches = resolution / self.patch_size
        if torch.any(num_patches != num_patches.int()):
            raise ValueError(f'The input resolution must be divisible by the patch size. Any of {resolution} cannot be divided by {self.patch_size}.')
        last_patch_idx = ((num_patches - 1) * self.patch_size).long()
        
        # Cyclic shift
        if self.use_cyclic_shift:
            left_split = x[:, :, :last_patch_idx[0], :, :]
            right_split = x[:, :, last_patch_idx[0]:, :, :]
            x = torch.cat([right_split, left_split], dim=2)  # ...| Cyclic shift X
            
            left_split = x[:, :, :, :last_patch_idx[1], :]
            right_split = x[:, :, :, last_patch_idx[1]:, :]
            x = torch.cat([right_split, left_split], dim=3)  # ...| Cyclic shift Y
            
            left_split = x[:, :, :, :, :last_patch_idx[2]]
            right_split = x[:, :, :, :, last_patch_idx[2]:]
            x = torch.cat([right_split, left_split], dim=4)  # ...| Cyclic shift Z
        
        # Multi-head self-attention
        x = x.unfold(dimension=2, size=self.patch_size, step=self.patch_size)  # ...| B, C, Nx, Ry, Rz, P       (unfold X)
        x = x.unfold(dimension=3, size=self.patch_size, step=self.patch_size)  # ...| B, C, Nx, Ny, Rz, P, P    (unfold Y)
        x = x.unfold(dimension=4, size=self.patch_size, step=self.patch_size)  # ...| B, C, Nx, Ny, Nz, P, P, P (unfold Z)
        backup_size = x.shape
        x = x.flatten(start_dim=2, end_dim=4)  # ...................................| B, C, N, P, P, P
        x = x.permute(0, 2, 1, 3, 4, 5)  # .........................................| B, N, C, P, P, P
        x = x.flatten(start_dim=2, end_dim=5)  # ...................................| B, N, CP3
        
        x, _ = self.attn(x, x, x)  # ...............................................| Self-attention
        x = x.view(batch_size, -1, self.num_channels, 
                   self.patch_size, self.patch_size, self.patch_size)  # ...........| B, N, C, P, P, P
        x = x.permute(0, 2, 1, 3, 4, 5)  # .........................................| B, C, N, P, P, P
        x = x.view(*backup_size)  # ................................................| B, C, Nx, Ny, Nz, P, P, P
        x = torch.cat(tuple(x.permute(7, *range(x.ndim - 1))), dim=4)  # ...........| B, C, Nx, Ny, Rz, P, P    (fold Z)
        x = torch.cat(tuple(x.permute(6, *range(x.ndim - 1))), dim=3)  # ...........| B, C, Nx, Ry, Rz, P       (fold Y)
        x = torch.cat(tuple(x.permute(5, *range(x.ndim - 1))), dim=2)  # ...........| B, C, Rx, Ry, Rz          (fold X)
        
        # Reverse cyclic shift
        if self.use_cyclic_shift:
            left_split = x[:, :, :self.patch_size, :, :]
            right_split = x[:, :, self.patch_size:, :, :]
            x = torch.cat([right_split, left_split], dim=2)  # ...| Reverse cyclic shift X
            
            left_split = x[:, :, :, :self.patch_size, :]
            right_split = x[:, :, :, self.patch_size:, :]
            x = torch.cat([right_split, left_split], dim=3)  # ...| Reverse cyclic shift Y
            
            left_split = x[:, :, :, :, :self.patch_size]
            right_split = x[:, :, :, :, self.patch_size:]
            x = torch.cat([right_split, left_split], dim=4)  # ...| Reverse cyclic shift Z
        
        if squeeze_later:
            return x.squeeze(0)
        else:
            return x
        

def test():
    gpu = 0
    torch.cuda.set_device(gpu)
    
    net = WindowAttention(6, 2, num_heads=8).cuda()
    sample_x = torch.randn(32, 6, 10, 10, 10).cuda()
    sample_y = net(sample_x)
    
    assert sample_x.shape == sample_y.shape
    print('PASS:', __file__)
    

if __name__ == '__main__':
    test()
