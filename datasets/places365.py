import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


class Places365(Dataset):
    """
    A dataset wrapper for existing Places365 data.
    Can be used with the pre-downloaded data, without requiring an additional download of the massive dataset.
    """
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
        return (img * 2.0) - 1.0  # Convert the image to the range [-1, 1]
