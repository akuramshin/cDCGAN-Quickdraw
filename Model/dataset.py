import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import h5py 
import numpy as np

class HDF5Dataset(Dataset):


    def __init__(self, h5_file, transform=None):
        super(HDF5Dataset, self).__init__()
        self.data_full = h5py.File(h5_file, 'r') 
        self.data = torch.from_numpy(np.array(self.data.get('X'))).float()
        self.target = torch.from_numpy(np.array(self.data.get('y'))).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
