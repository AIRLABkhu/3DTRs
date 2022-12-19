import torch
from torch import nn


class SparseConv3d(nn.Module):
    def __init__(self, resolution: int, in_channels: int, out_channels: int, kernel_size: int, bias: bool=False):
        super(SparseConv3d, self).__init__()
        
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.use_bias = bias
        
        self.weight = nn.Parameter(torch.randn(self.out_channels, self.in_channels, kernel_size, kernel_size, kernel_size), requires_grad=True)
        if self.use_bias:
            self.bias = nn.Parameter(torch.randn(self.out_channels), requires_grad=True)
        else:
            self.bias = nn.Parameter(None, requires_grad=False)

        left_pad_size = kernel_size // 2
        right_pad_size = kernel_size - 1 - left_pad_size
        kernel_offset = tuple(range(-left_pad_size, right_pad_size + 1))
        kernel_offset_perm = []
        for i in kernel_offset:
            for j in kernel_offset:
                for k in kernel_offset:
                    kernel_offset_perm.append((i, j, k))
        self.kernel_offset = torch.tensor(kernel_offset_perm)
            
    def forward(self, x: torch.Tensor):
        batch_size, in_channels, *resolution = x.shape
        resolution = torch.tensor(resolution)
        
        pad_size = self.kernel_size // 2
        padded_resolution = resolution + (self.kernel_size & ~1)
        pad_x = torch.zeros(batch_size, in_channels, pad_size, resolution[1], resolution[2]).to(x.device)
        pad_y = torch.zeros(batch_size, in_channels, padded_resolution[0], pad_size, resolution[2]).to(x.device)
        pad_z = torch.zeros(batch_size, in_channels, padded_resolution[0], padded_resolution[1], pad_size).to(x.device)
        
        x = torch.cat([pad_x, x, pad_x], dim=2)
        x = torch.cat([pad_y, x, pad_y], dim=3)
        x = torch.cat([pad_z, x, pad_z], dim=4)
            
        result = torch.zeros(batch_size, self.out_channels, *resolution, dtype=torch.float, device=x.device)
        for sample_idx, sample in enumerate(x):  # ......................................................................| CUDA
            sample_nonzero_indices = sample.cpu().abs().sum(dim=0).nonzero()  # .........................................| CPU
            num_nonzeros = sample_nonzero_indices.size(0)
            indices = sample_nonzero_indices.repeat(1, self.kernel_size ** 3).view(-1, 3)
            kernel_offset = self.kernel_offset.repeat(num_nonzeros, 1).to(indices.device)
            
            kernel_indices = (indices + kernel_offset).reshape(num_nonzeros, -1, 3) \
                .permute(2, 0, 1).flatten(start_dim=1)
            slice_x, slice_y, slice_z = kernel_indices
            
            sample = sample[:, slice_x, slice_y, slice_z] \
                .view(1, num_nonzeros, in_channels, self.kernel_size, self.kernel_size, self.kernel_size) \
                .expand(self.out_channels, -1, -1, -1, -1, -1)  # .......................................................| CUDA
            weight = self.weight \
                .unsqueeze(1) \
                .expand(-1, num_nonzeros, -1, -1, -1, -1)
            product = (sample * weight).sum(dim=(-1, -2, -3, -4)).flatten()
                
            result_indices = sample_nonzero_indices \
                .unsqueeze(0).expand(self.out_channels, -1, -1)  # ......................................................| CPU
            result_indices = torch.cat([
                result_indices - 1,
                torch.cat([
                    torch.full(size=(1, result_indices.size(1), 1), fill_value=i)
                    for i in range(self.out_channels)], dim=0)
            ], dim=2)
            result_indices = result_indices.permute(2, 0, 1).flatten(start_dim=1)
            
            slice_x, slice_y, slice_z, slice_c = result_indices  # ......................................................| CUDA
            result[sample_idx, slice_c, slice_x, slice_y, slice_z] = product

        return result


def test(device='cuda:6', large_input=False, vis=False):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    if large_input:
        batch_size=32
        resolution=30
        in_channels=6
        out_channels=32 
    else:
        batch_size = 3
        resolution = 10
        in_channels = 2
        out_channels = 4
    
    kernel_size = 3
    bias = False

    conv = SparseConv3d(resolution, in_channels, out_channels, kernel_size, bias).to(device)
    x = torch.rand(batch_size, in_channels, resolution, resolution, resolution).to(device)
    x = ((x < 0.1) * x * 100).int() * 0.1  # sparsify
    y = conv(x)

    assert y.shape == (batch_size, out_channels, resolution, resolution, resolution)
    print('PASS:', __file__)
    
    if vis:
        sample_x = x[0, 0].detach().cpu()  # res, res, res
        sample_y = y[0, 0].detach().cpu()  # res, res, res
        
        nz_x = sample_x.nonzero().permute(1, 0)
        nz_y = sample_y.nonzero().permute(1, 0)
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 4))
        
        ax = plt.subplot(1, 3, 1, projection='3d')
        plt.title('Original')
        ax.scatter(nz_x[0], nz_x[1], nz_x[2], c=sample_x[nz_x[0], nz_x[1], nz_x[2]], cmap='jet')
        ax.set_xlim((-1, resolution + 1))
        ax.set_ylim((-1, resolution + 1))
        ax.set_zlim((-1, resolution + 1))
        plt.jet(); plt.contourf(sample_x[0], sample_x[1], sample_x[2]); plt.colorbar()
        
        ax = plt.subplot(1, 3, 2, projection='3d')
        plt.title('After SPConv')
        ax.scatter(nz_y[0], nz_y[1], nz_y[2], c=sample_y[nz_y[0], nz_y[1], nz_y[2]], cmap='jet')
        ax.set_xlim((-1, resolution + 1))
        ax.set_ylim((-1, resolution + 1))
        ax.set_zlim((-1, resolution + 1))
        plt.jet(); plt.contourf(sample_y[0], sample_y[1], sample_y[2]); plt.colorbar()
        
        ax = plt.subplot(1, 3, 3, projection='3d')
        plt.title('Changes')
        changes = torch.abs(sample_x - sample_y)
        nz_c = changes.nonzero().permute(1, 0)
        ax.scatter(nz_c[0], nz_c[1], nz_c[2], c=changes[nz_c[0], nz_c[1], nz_c[2]], cmap='jet')
        ax.set_xlim((-1, resolution + 1))
        ax.set_ylim((-1, resolution + 1))
        ax.set_zlim((-1, resolution + 1))
        plt.jet(); plt.contourf(changes[0], changes[1], changes[2]); plt.colorbar()
        
        plt.savefig('temp/temp.png')


def benchmark(device='cuda:0', seed=None, num_batches=1000, batch_size=32, resolution=30, in_channels=6, out_channels=32, kernel_size=3, bias=False):
    import timeit
    
    if seed is not None:
        if device.lower().startswith('cuda'):
            torch.cuda.set_device(device)
            torch.cuda.manual_seed(device)
        torch.manual_seed(seed)
    
    conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias).to(device)
    spconv = SparseConv3d(resolution, in_channels, out_channels, kernel_size, bias).to(device)
    
    x = torch.rand(batch_size, in_channels, resolution, resolution, resolution).to(device)
    x = ((x < 0.1) * x * 100).int() * 0.1  # sparsify    
        
    def run_conv():
        y = conv(x) 
    
    def run_spconv():
        y = spconv(x)

    time_conv = timeit.timeit(stmt=run_conv, number=num_batches)
    print('Run nn.Conv3D %d times: %.4fs' % (num_batches, time_conv))
    
    time_spconv = timeit.timeit(stmt=run_spconv, number=num_batches)
    print('Run  SPConv3D %d times: %.4fs' % (num_batches, time_spconv))


if __name__ == '__main__':
    test(large_input=False, vis=True)
    benchmark()
