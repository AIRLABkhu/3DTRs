import torch


def cyclic_shift_3d(x: torch.Tensor, patch_size: int, inverse: bool=False):
    if not inverse:
        patch_indices = torch.tensor(tuple(x.shape)[2:]) - patch_size
        
        left_split = x[:, :, :patch_indices[0], :, :]
        right_split = x[:, :, patch_indices[0]:, :, :]
        x = torch.cat([right_split, left_split], dim=2)  # ...| Cyclic shift X
        
        left_split = x[:, :, :, :patch_indices[1], :]
        right_split = x[:, :, :, patch_indices[1]:, :]
        x = torch.cat([right_split, left_split], dim=3)  # ...| Cyclic shift Y
        
        left_split = x[:, :, :, :, :patch_indices[2]]
        right_split = x[:, :, :, :, patch_indices[2]:]
        x = torch.cat([right_split, left_split], dim=4)  # ...| Cyclic shift Z
    else:
        left_split = x[:, :, :patch_size, :, :]
        right_split = x[:, :, patch_size:, :, :]
        x = torch.cat([right_split, left_split], dim=2)  # ...| Reverse cyclic shift X
        
        left_split = x[:, :, :, :patch_size, :]
        right_split = x[:, :, :, patch_size:, :]
        x = torch.cat([right_split, left_split], dim=3)  # ...| Reverse cyclic shift Y
        
        left_split = x[:, :, :, :, :patch_size]
        right_split = x[:, :, :, :, patch_size:]
        x = torch.cat([right_split, left_split], dim=4)  # ...| Reverse cyclic shift Z
    
    return x
