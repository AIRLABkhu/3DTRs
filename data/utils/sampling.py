from abc import ABC, abstractmethod

import torch


class Sampler(ABC):
    def __init__(self, num_points: int):
        self.num_points = num_points
        
    @abstractmethod
    def sample(self, x):
        pass
    
    def __call__(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(0)
        
        return self.sample(x)
    
    
class UniformSampler(Sampler):
    def sample(self, x):
        '''
        def farthest_point_sample(xyz, npoint):
        https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
        '''
        device = x.device
        B, N, _ = x.shape
        centroids = torch.zeros(B, self.num_points, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(self.num_points):
            centroids[:, i] = farthest
            centroid = x[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((x - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        
        return centroids


class RandomSampler(Sampler):
    def sample(self, x):
        return torch.randperm(x.size(1)).to(self.device)
    

class IdentitySampler(Sampler):
    def sample(self, x):
        return torch.arange(x.size(1)).to(self.device)


def get_sampler(name: str, num_points):
    return {
        'none': IdentitySampler,
        'random': RandomSampler,
        'uniform': UniformSampler,
    }[name](num_points=num_points)
