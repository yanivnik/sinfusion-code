import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RandomScaleResize(object):
    def __init__(self, min_scale=0.5, max_scale=1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, x):
        if random.random() < 0.2:  # TODO CONVERT TO CONSTANT IN CONFIG
            return x
        scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
        resized_size = (int(x.size[0] * scale), int(x.size[0] * scale))
        return transforms.functional.resize(x, resized_size)


class CropSet(Dataset):
    """
    A dataset comprised of crops of various augmentation of a single image.
    """
    def __init__(self, image_path, crop_size=(64, 64)):
        self.crop_size = crop_size

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            RandomScaleResize(),
            transforms.RandomCrop(self.crop_size, pad_if_needed=True, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: (img[:3, ] * 2) - 1)
        ])

        self.img = Image.open(image_path)

    def __len__(self):
        return 5000  # TODO CHANGE THIS? THIS SHOULD BE A HIGH NUMBER BECAUSE OTHERWISE THERE IS A SHIT-TON OF OVERHEAD FROM PL

    def __getitem__(self, item):
        return self.transform(self.img)
