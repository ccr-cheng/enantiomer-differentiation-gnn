import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class EnantiomerDataset(Dataset):
    def __init__(self, root, split, rotate=True):
        self.root = root
        self.split = split
        self.rotate = rotate

        data = np.load(f'{root}/{split}.npz')
        self.z = torch.LongTensor(data['z'])
        self.pos = data['pos']
        self.label = torch.FloatTensor(data['label'])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        pos = self.pos[item]
        if self.rotate:
            rot = R.random()
            pos = rot.apply(pos)
        data = Data(z=self.z, pos=torch.FloatTensor(pos), y=self.label[item])
        return data


def get_dataset(cfg):
    d_cfg = cfg.copy()
    train_cfg = d_cfg.pop('train', {})
    test_cfg = d_cfg.pop('test', {})
    return (
        EnantiomerDataset(split='train', **train_cfg, **d_cfg),
        EnantiomerDataset(split='test', **test_cfg, **d_cfg),
    )
