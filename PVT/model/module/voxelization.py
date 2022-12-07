import torch
from torch import nn


class Voxelization(nn.Module):
    def __init__(self, resolution: int, eps=1.0E-8):
        super(Voxelization, self).__init__()
        
        self.resolution = resolution
        self.eps = eps
        
    def forward(self, pts, feat=None):
        if feat is None:
            feat = pts
        
        batch_size, _, num_points = pts.shape
        _, feat_dim, _ = feat.shape
        
        with torch.no_grad():
            pts_detached = pts.clone().detach()
            
            pts_normalized = pts_detached - pts_detached.min(dim=2, keepdim=True).values.expand_as(pts)
            pts_normalized = pts_normalized / (pts_normalized.norm(dim=1).max() + self.eps)
            
            vox_indices_3d = pts_normalized * (self.resolution - 1)
            vox_indices_3d = vox_indices_3d.long()
            
            vox_indices = vox_indices_3d[:, 0, :] + \
                          vox_indices_3d[:, 1, :] * self.resolution + \
                          vox_indices_3d[:, 2, :] * self.resolution * self.resolution
            pts_indices = torch.arange(num_points).to(pts.device)
            pts_indices = pts_indices.unsqueeze(0).expand(batch_size, num_points)
            
        voxel = torch.zeros(batch_size, self.resolution ** 3, feat_dim).float().to(feat.device)
        vox_indices_sort_values, vox_indices_sort_indices = torch.sort(vox_indices, dim=1)
        for sample_idx, (sample_value, sample_vox_indices, sample_pts_indices, sample_feat) \
            in enumerate(zip(vox_indices_sort_values, vox_indices_sort_indices, pts_indices, feat)):
                
            unique_at = torch.unique(sample_value, return_counts=True)[1]
            unique_indices = torch.cumsum(unique_at, dim=0)
            unique_indices = [0] + unique_indices.tolist()
            
            for start, end in zip(unique_indices[:-1], unique_indices[1:]):
                pts_idx_span = sample_pts_indices[start:end]
                feature_cell = sample_feat[:, pts_idx_span].mean(dim=1)
                
                vox_idx_idx = sample_vox_indices[start]
                vox_idx = vox_indices[sample_idx, vox_idx_idx]
                
                voxel[sample_idx, vox_idx] = feature_cell

        return voxel.permute(0, 2, 1).reshape(batch_size, feat_dim, self.resolution, self.resolution, self.resolution)


def test():
    device = 'cuda'
    x = torch.cat([
        torch.randn(32, 3, 256).cuda(),
        torch.randn(32, 3, 256).cuda() + 3,
        torch.randn(32, 3, 256).cuda() - 4,
        torch.randn(32, 3, 256).cuda() + 7,
    ], dim=2).to(device)
    f = torch.randn(x.size(0), 6, x.size(2), requires_grad=True).to(device)
    f.retain_grad()

    out = Voxelization(30).to(device)(x, f)
    assert out.shape == (32, 6, 30, 30, 30)

    s = out.sum()
    s.backward()
    assert f.grad.shape == f.shape
    
if __name__ == '__main__':
    test()
    print('PASSED:', __file__)
