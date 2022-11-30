import os
from pathlib import Path
from tqdm.auto import tqdm
import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


class PartNetDataLoader(Dataset):
    DEFAULT_ROOT = '/material/data/PartNet/sem_seg_h5'
    
    def __init__(self, root: str, folder: str, split: str='train', num_points: int=2500):
        self.root = root
        self.folder = folder
        self.split = split
        
        self.num_points = num_points
        
        # Check integrity
        available_folders = os.listdir(root)
        if folder not in available_folders:
            raise ValueError('Folder must be one of (' + ', '.join(available_folders) + ').')
        
        # Read file list
        h5_folder = Path(root).joinpath(folder) # ................................| root/Bag-1
        list_filename = h5_folder.joinpath(f'{self.split}_files.txt') # ..........| root/Bag-1/train_files.txt
        with open(list_filename, 'r') as list_file:
            h5_filenames = list_file.readlines() # ...............................| [train-00.h5, ...]
        
        # Load dataset
        points, masks = [], []
        desc = f'LOADING PARTNET_{self.folder.upper()}_{self.split.upper()}' # ...| train-00.h5
        for filename in tqdm(h5_filenames, desc=desc): 
            h5_fullname = h5_folder.joinpath(filename[2:-1]) # ...................| root/Bag-1/train-00.h5
            with h5py.File(h5_fullname) as h5_file:
                point_batch = torch.tensor(np.array(h5_file['data'])).float()
                mask_batch = torch.tensor(np.array(h5_file['label_seg'])).long()
                
                points.append(point_batch)
                masks.append(mask_batch)
        self.points = torch.cat(points, dim=0)
        self.masks = torch.cat(masks, dim=0)
        
    def __getitem__(self, idx):
        return self.points[idx], self.masks[idx]
    
    def __len__(self):
        return self.points.size(0)


if __name__ == '__main__':
    dataset = PartNetDataLoader(PartNetDataLoader.DEFAULT_ROOT, folder='Chair-2')
    print(len(dataset))
    
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for i, (points, masks) in enumerate(loader):
        if i == 3:
            break
