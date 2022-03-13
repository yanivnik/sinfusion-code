import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


class Places365(Dataset):
    def __init__(self, train, root='./data'):
        transform = transforms.Compose([transforms.ToTensor()])

        self.dataset = ImageFolder(os.path.join(root, 'train' if train else 'val'), transform=transform)

        self.dataset_len = len(self.dataset)
        self.depth = 3
        self.size = 256

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        img = self.dataset[item][0]
        return (((img / 255.0) * 2.0) - 1.0).moveaxis(3, 1)