import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

from importlib import import_module



class S3DISDataset(Dataset):
    DEFAULT_ROOT = '/material/data/s3dis'
    
    def __init__(self, split='train', data_root='trainval_fullarea', mask_mode='semantic', num_point=4096, test_area=5, block_size=1.0, sample_rate=1.0, transform=None):
        super().__init__()
        self.mask_mode = mask_mode
        if mask_mode not in ('semantic', 'instance'):
            raise ValueError('Invalid mask mode. Must be semantic or instance.')
        
        self.num_point = num_point
        self.block_size = block_size
        self.transform = transform
        rooms = sorted(os.listdir(data_root))
        rooms = [room for room in rooms if 'Area_' in room]
        if split == 'train':
            rooms_split = [room for room in rooms if not 'Area_{}'.format(test_area) in room]
        else:
            rooms_split = [room for room in rooms if 'Area_{}'.format(test_area) in room]

        self.room_points, self.room_labels = [], []
        self.room_coord_min, self.room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)

        i = 0
        for room_name in tqdm(rooms_split, total=len(rooms_split)):
            
            room_path = os.path.join(data_root, room_name)
            room_data = np.load(room_path, allow_pickle=True)  # xyzrgbl, N*7
            
            for data in room_data:
                i += 1
                pts_path = os.path.join(data_root, data['pts_path'])
                mask_path = os.path.join(data_root, data[f'pts_{self.mask_mode}_mask_path'])
                
                points = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 6)
                labels = np.fromfile(mask_path, dtype=np.int64).reshape(-1, 1)
            
                # points, labels = room_data[:, 0:6], room_data[:, 6]  # xyzrgb, N*6; l, N
                tmp, _ = np.histogram(labels, range(14))
                labelweights += tmp
                coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
                self.room_points.append(points), self.room_labels.append(labels)
                self.room_coord_min.append(coord_min), self.room_coord_max.append(coord_max)
                num_point_all.append(labels.size)
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print(self.labelweights)
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) * sample_rate / num_point)
        room_idxs = []
        for index in range(len(rooms_split)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        self.room_idxs = np.array(room_idxs)
        print("Totally {} samples in {} set.".format(len(self.room_idxs), split))

    def __getitem__(self, idx):
        room_idx = self.room_idxs[idx]
        points = self.room_points[room_idx]   # N * 6
        labels = self.room_labels[room_idx]   # N
        N_points = points.shape[0]

        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [self.block_size / 2.0, self.block_size / 2.0, 0]
            block_max = center + [self.block_size / 2.0, self.block_size / 2.0, 0]
            point_idxs = np.where((points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break

        if point_idxs.size >= self.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / self.room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / self.room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / self.room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]
        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.room_idxs)

if __name__ == '__main__':
    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01

    point_data = S3DISDataset(split='train', data_root=S3DISDataset.DEFAULT_ROOT, num_point=num_point, test_area=test_area, block_size=block_size, sample_rate=sample_rate, transform=None)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()