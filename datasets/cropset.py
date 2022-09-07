import random
from builtins import range

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CropSet(Dataset):
    """
    A dataset comprised of crops of various augmentation of a single image.
    """
    def __init__(self, image, crop_size, dataset_size=5000):
        """
        Args:
            image (torch.tensor): The image to generate crops from.
            crop_size (tuple(int, int)): The spatial dimensions of the crops to be taken.
            dataset_size (int): The amount of images in a single epoch of training. For training datasets,
                                this should be a high number to avoid overhead from pytorch_lightning.
        """
        self.crop_size = crop_size
        self.dataset_size = dataset_size

        self.transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(self.crop_size, pad_if_needed=False),
            transforms.Lambda(lambda img: (img[:3, ] * 2) - 1)
        ])

        self.img = image

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        if len(self.img.shape) == 3:
            return {'IMG': self.transform(self.img)}
        else:
            frame_idx = random.randrange(0, self.img.shape[0])
            return {'IMG': self.transform(self.img[frame_idx]), 'FRAME': frame_idx}
