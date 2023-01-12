'''
2022. 08. 22. 유근혁
Semantic segmentation을 nuScenes 데이터셋.
SegmentationDataset 클래스 구현 및 사용법.
'''

import os
from tqdm.auto import tqdm
from typing import Any, Union
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


class SegmentationDataset(Dataset):
    def __init__(
        self, version: str='v1.0-mini', dataroot: str='/data/sets/nuscenes', verbose: bool=False, 
        transforms: Any=None, num_points: int=None, train: bool=True, split_rate: Union[None, float]=None):

        if not split_rate:
            split_rate = 0.7 if train else 0.3
        if not 0 < split_rate <= 1:
            raise RuntimeError('split_rate must be in (0, 1].')
        self.split_rate = split_rate
        
        self.__version = version
        self.__dataroot = dataroot

        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        self.transforms = transforms
        self.train = train

        self.verbose = verbose

        min_num_points = torch.inf
        for i in range(len(self)):
            lidarseg = self.nusc.lidarseg[i]
            sample_data = self.nusc.get('sample_data', lidarseg['sample_data_token'])
            filename = os.path.join(self.dataroot, sample_data['filename'])
            pc = torch.from_numpy(LidarPointCloud.from_file(filename).points)

            n = pc.size(-1)
            min_num_points = min(n, min_num_points)
        if not num_points or num_points > min_num_points:
            self.num_points = min_num_points
        else:
            self.num_points = num_points

    def __len__(self):
        return int(len(self.nusc.lidarseg) * self.split_rate)

    def __getitem__(self, idx):
        num_lidarseg_labels = len(self.nusc.lidarseg_idx2name_mapping.keys())

        def load_at(i: int):
            lidarseg = self.nusc.lidarseg[i]

            label = np.fromfile(os.path.join(self.dataroot, lidarseg['filename']), dtype=np.uint8)
            label = torch.from_numpy(label)
            onehot = torch.zeros(label.size(0), num_lidarseg_labels)
            onehot = onehot.scatter_(1, label.unsqueeze(0).long(), 1)

            sample_data = self.nusc.get('sample_data', lidarseg['sample_data_token'])
            lidar_filename = os.path.join(self.dataroot, sample_data['filename'])
            lidar_pc = torch.from_numpy(LidarPointCloud.from_file(lidar_filename).points)

            indices = torch.linspace(0, lidar_pc.size(-1) - 1, self.num_points).round().long()
            lidar_pc = lidar_pc[:, indices]
            label = label[indices]
            onehot = onehot[indices, :]

            return lidar_pc, label, onehot

        if type(idx) is int:
            if not self.train:
                idx = -idx

            pc, label, onehot = load_at(idx)
            return pc, label, onehot
        else:
            def if_none(i, val):
                return i if i else val

            start = if_none(idx.start, 0)
            stop = if_none(idx.stop, len(self))
            step = if_none(idx.step, 1)
            if not self.train:
                total = len(self.nusc.lidarseg)
                start, stop, step = total - start - 1, total - stop - 1, -step
            
            pcs, labels, onehots = [], [], []
            for i in range(start, stop, step):
                pc, label, onehot = load_at(i)

                pcs.append(pc.unsqueeze(0))
                labels.append(label.unsqueeze(0))
                onehots.append(onehot.unsqueeze(0))
            
            return torch.cat(pcs, dim=0), torch.cat(labels, dim=0), torch.cat(onehots, dim=0)

    def __get_dataroot(self):
        return self.__dataroot

    def __get_version(self):
        return self.__version

    dataroot = property(__get_dataroot)
    version = property(__get_version)


if __name__ == '__main__':
    DATAROOT = '../nuScenes/data/sets/nuscenes'.replace('/', os.sep)
    DATAROOT = os.path.abspath(DATAROOT)

    train_dataset = SegmentationDataset(dataroot=DATAROOT, split_rate=0.7, train=True)
    test_dataset = SegmentationDataset(dataroot=DATAROOT, split_rate=0.3, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
