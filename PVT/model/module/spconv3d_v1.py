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
        
        self.weight = nn.Parameter(torch.randn(1, self.out_channels, self.in_channels, kernel_size, kernel_size, kernel_size), requires_grad=True)
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
        num_channels, resolution = x.size(1), torch.tensor(tuple(x.shape[2:]))
        
        pad_size = self.kernel_size // 2
        padded_resolution = resolution + (self.kernel_size & ~1)
        pad_x = torch.zeros(num_channels, pad_size, resolution[1], resolution[2]).to(x.device)
        pad_y = torch.zeros(num_channels, padded_resolution[0], pad_size, resolution[2]).to(x.device)
        pad_z = torch.zeros(num_channels, padded_resolution[0], padded_resolution[1], pad_size).to(x.device)
        
        nonzero_indices = x.cpu().nonzero()
        unique_values, unique_counts = nonzero_indices[:, 0].unique_consecutive(return_counts=True)
        unique_indices = [0] + unique_counts.cumsum(dim=0).tolist()
        
        results = []
        for sample_idx, sample, sample_start, sample_end in zip(unique_values, x, unique_indices[:-1], unique_indices[1:]):
            sample_nonzero_indices = nonzero_indices[sample_start:sample_end, 2:]  # except channel axis

            num_nonzeros = sample_nonzero_indices.size(0)
            indices = sample_nonzero_indices.repeat(1, self.kernel_size ** 3).view(-1, 3)
            kernel_offset = self.kernel_offset.repeat(num_nonzeros, 1)
            indices = (indices + kernel_offset).reshape(num_nonzeros, -1, 3).permute(2, 0, 1).flatten(start_dim=1)
            slice_x, slice_y, slice_z = indices
            
            padded_sample = sample
            padded_sample = torch.cat([pad_x, padded_sample, pad_x], dim=1)
            padded_sample = torch.cat([pad_y, padded_sample, pad_y], dim=2)
            padded_sample = torch.cat([pad_z, padded_sample, pad_z], dim=3)
            
            sliced_sample = padded_sample[:, slice_x, slice_y, slice_z]
            sliced_sample = sliced_sample.view(1, num_channels, num_nonzeros, 
                                               self.kernel_size, self.kernel_size, 
                                               self.kernel_size) \
                                         .expand(self.out_channels, -1, -1, -1, -1, -1) \
                                         .permute(2, 0, 1, 3, 4, 5)                             
            # self.weight: 1, out, in, ker, ker, ker
            kernel = self.weight.expand(num_nonzeros, -1, -1, -1, -1, -1)
            
            # both: nz, out, in, ker, ker, ker
            flattened_sample = sliced_sample.flatten(start_dim=0)
            flattened_kernel = kernel.flatten(start_dim=0)
            
            local_indices = []
            i = 0
            while i < flattened_sample.size(0):
                next_i = i + 2_147_483_647 - 1  # INT_MAX
                flattened_sample_part = flattened_sample[i:next_i]
                local_indices.append(flattened_sample_part.nonzero() + i)
                i = next_i
            local_indices = torch.cat(local_indices, dim=0)
            
            nonzero_sample = flattened_sample[local_indices]
            nonzero_kernel = flattened_kernel[local_indices]
            nonzero_product = nonzero_sample * nonzero_kernel
            
            result = torch.zeros(1, self.out_channels, *resolution).float()
            local_indices_vox = sliced_sample.nonzero()
            _, local_indices_unique_counts = local_indices_vox[:, 0].unique_consecutive(dim=0, return_counts=True)
            local_indices_unique_indices = [0] + (local_indices_unique_counts.cumsum(dim=0) * self.out_channels).tolist()
            for vox_at, cell_start, cell_end in zip(sample_nonzero_indices, 
                                                    local_indices_unique_indices[:-1], 
                                                    local_indices_unique_indices[1:]):
                sliced_product = nonzero_product[cell_start:cell_end].view(-1, self.out_channels).sum(dim=0)
                result[0, :, vox_at[0], vox_at[1], vox_at[2]] = sliced_product
            results.append(result)
            
        # result: batch, out, res, res, res
        # self.bias: out
        result = torch.cat(results, dim=0).to(x.device)
        if self.use_bias:
            bias = self.bias.view(1, -1, 1, 1, 1).expand_as(result)
            return result + bias
        else:
            return result


def test(device='cuda:6', vis=False):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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
        plt.jet(); plt.contourf(sample_x[0], sample_x[1], sample_x[2]); plt.colorbar()
        
        ax = plt.subplot(1, 3, 2, projection='3d')
        plt.title('After SPConv')
        ax.scatter(nz_y[0], nz_y[1], nz_y[2], c=sample_y[nz_y[0], nz_y[1], nz_y[2]], cmap='jet')
        plt.jet(); plt.contourf(sample_y[0], sample_y[1], sample_y[2]); plt.colorbar()
        
        ax = plt.subplot(1, 3, 3, projection='3d')
        plt.title('Changes')
        changes = torch.abs(sample_x - sample_y)
        ax.scatter(nz_x[0], nz_x[1], nz_x[2], c=changes[nz_x[0], nz_x[1], nz_x[2]], cmap='jet')
        plt.jet(); plt.contourf(changes[0], changes[1], changes[2]); plt.colorbar()
        
        plt.savefig('temp/temp.png')


def benchmark(device='cuda:6', seed=None, num_batches=10, batch_size=32, resolution=30, in_channels=6, out_channels=32, kernel_size=3, bias=False):
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
    
    def run_spconv():
        y = spconv(x)
        del y
        
    def run_conv():
        y = conv(x)
        del y

    time_spconv = timeit.timeit(stmt=run_spconv, number=num_batches)
    print('Run  SPConv3D %d times: %.4fs' % (num_batches, time_spconv))
    
    time_conv = timeit.timeit(stmt=run_conv, number=num_batches)
    print('Run nn.Conv3D %d times: %.4fs' % (num_batches, time_conv))


if __name__ == '__main__':
    test(vis=True)
    # benchmark()
