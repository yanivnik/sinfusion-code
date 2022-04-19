import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class RandomScaleResize(object):
    def __init__(self, min_scale=0.7, max_scale=1.0):
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

        # TODO: Think about fine tuning these transformations.
        #   Maybe a rotation that happens all the time isn't a good idea (learn from zssr.random_augment).
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation((0, 180), transforms.InterpolationMode.BICUBIC),
            RandomScaleResize(),
            transforms.RandomCrop(self.crop_size, pad_if_needed=True, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: (img[:3, ] * 2) - 1)
        ])

        self.img = Image.open(image_path)
        # self.img = transforms.ToTensor()(self.img)
        # self.img = ((self.img * 2.0) - 1.0)

        # c, h, w = self.img.shape
        # self.dataset_len = L
        # self.depth = c
        # self.size = h

    def __len__(self):
        return 5000  # TODO CHANGE THIS? THIS SHOULD BE A HIGH NUMBER BECAUSE OTHERWISE THERE IS A SHIT-TON OF OVERHEAD FROM PL

    def __getitem__(self, item):
        return self.transform(self.img)
