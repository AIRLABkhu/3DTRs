import torch
from torch import nn


class Devoxelization(nn.Module):
    def __init__(self, resolution: int, eps=1.0E-8):
        super(Devoxelization, self).__init__()
        
        self.resolution = resolution
        self.eps = eps
        
    def forward(self, pts, feat):
        with torch.no_grad():
            pts_detached = pts.clone().detach()
            
            pts_normalized = pts_detached - pts_detached.min(dim=2, keepdim=True).values.expand_as(pts)
            pts_normalized = pts_normalized / (pts_normalized.norm(dim=1).max() + self.eps)
            
            vox_indices_3d = pts_normalized * (self.resolution - 1)  # ...| for right element
            vox_indices_3d_C = 1 - vox_indices_3d  # .....................| for left element, 1s' complementary
            vox_indices_3d_left = torch.floor(vox_indices_3d).long()
            vox_indices_3d_right = torch.ceil(vox_indices_3d).long()
            
        new_feat = []
        for sample_feat, sample_idx_left, sample_idx_right, sample_weight_left, sample_weight_right \
            in zip(feat, vox_indices_3d_left, vox_indices_3d_right, vox_indices_3d_C, vox_indices_3d):
            feat000 = sample_feat[:, sample_idx_left[0],  sample_idx_left[1],  sample_idx_left[2]]
            feat100 = sample_feat[:, sample_idx_right[0], sample_idx_left[1],  sample_idx_left[2]]
            feat010 = sample_feat[:, sample_idx_left[0],  sample_idx_right[1], sample_idx_left[2]]
            feat110 = sample_feat[:, sample_idx_right[0], sample_idx_right[1], sample_idx_left[2]]
            feat001 = sample_feat[:, sample_idx_left[0],  sample_idx_left[1],  sample_idx_right[2]]
            feat101 = sample_feat[:, sample_idx_right[0], sample_idx_left[1],  sample_idx_right[2]]
            feat011 = sample_feat[:, sample_idx_left[0],  sample_idx_right[1], sample_idx_right[2]]
            feat111 = sample_feat[:, sample_idx_right[0], sample_idx_right[1], sample_idx_right[2]]
            
            feat_00 = feat000 * sample_weight_left[0] + feat100 * sample_weight_right[0]
            feat_10 = feat010 * sample_weight_left[0] + feat110 * sample_weight_right[0]
            feat_01 = feat001 * sample_weight_left[0] + feat101 * sample_weight_right[0]
            feat_11 = feat011 * sample_weight_left[0] + feat111 * sample_weight_right[0]
            
            feat__0 = feat_00 * sample_weight_left[1] + feat_10 * sample_weight_right[1]
            feat__1 = feat_01 * sample_weight_left[1] + feat_11 * sample_weight_right[1]
            
            feat___ = feat__0 * sample_weight_left[2] + feat__1 * sample_weight_right[2]
            new_feat.append(feat___.unsqueeze(0))
        return torch.cat(new_feat, dim=0)
        
        
def test():
    device = 'cuda'
    x = torch.cat([
        torch.randn(32, 3, 256).cuda(),
        torch.randn(32, 3, 256).cuda() + 3,
        torch.randn(32, 3, 256).cuda() - 4,
        torch.randn(32, 3, 256).cuda() + 7,
    ], dim=2).to(device)
    vox = torch.randn(x.size(0), 6, 30, 30, 30, requires_grad=True).to(device)
    vox.retain_grad()

    out = Devoxelization(30).to(device)(x, vox)
    assert out.shape == (32, 6, 1024)

    s = out.sum()
    s.backward()
    assert vox.grad.shape == vox.shape
    
if __name__ == '__main__':
    test()
    print('PASSED:', __file__)

