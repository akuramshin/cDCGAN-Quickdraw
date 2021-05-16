import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class QuickdrawDataset(Dataset):


    def __init__(self, datapath, targetpath, transform=None):
        super(QuickdrawDataset, self).__init__()
        self.data = np.load(datapath)
        self.data = np.expand_dims(self.data, axis=1)
        self.target = torch.tensor(np.load(targetpath)).long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        y = torch.nn.functional.one_hot(y, 10) 

        if self.transform:
            img = Image.fromarray(x.reshape(28,28).astype(np.uint8), 'L')
            x = self.transform(img)
        
        return x, y
    
    def __len__(self):
        return len(self.data)
