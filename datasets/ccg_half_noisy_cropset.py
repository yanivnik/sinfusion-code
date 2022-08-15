import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.transforms import RandomScaleResize


class CCGSemiNoisyCropSet(Dataset):
    """
    A dataset comprised of pairs of augmented crops. Each crop pair includes a normal crop,
    and a second, semi-noisy crop. In this context "semi noisy" means that a specific chunk
    of the crop is complete noise, while the other is a normal image.

    The semi noisy crop is used as a conditioning signal to generate the normal crop, which should match it
    in the normal part.
    This is used for crop-conditional generation experiments.
    """
    def __init__(self, image, crop_size, noisy_chunk_size=None, dataset_size=5000):
        """
        Args:
            image (torch.tensor): The image to generate crops from.
            crop_size (tuple(int, int)): The spatial dimensions of the crops to be taken.
            noisy_chunk_size (tuple(int, int)): The size of the pure noise part in the semi-noisy crop.
                                                If None, half of the crop_size in each dimension is used.
            dataset_size (int): The amount of images in a single epoch of training. For training datasets,
                                this should be a high number to avoid overhead from pytorch_lightning.
        """
        self.crop_size = crop_size
        self.noisy_chunk_size = noisy_chunk_size or (self.crop_size[0] // 2, self.crop_size[1] // 2)
        self.dataset_size = dataset_size

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            RandomScaleResize(),
            transforms.RandomCrop(self.crop_size, pad_if_needed=True, padding_mode='constant'),
            transforms.Lambda(lambda img: (img[:3, ] * 2) - 1)
        ])

        self.img = image

    def _replace_with_noise(self, crop):
        half_noisy_crop = crop.clone()

        # Change a chunk of the image to be complete noise in a random location
        h_noise_index = random.randint(0, self.noisy_chunk_size[0] - 1)
        w_noise_index = random.randint(0, self.noisy_chunk_size[1] - 1)
        half_noisy_crop[:, h_noise_index:h_noise_index + self.noisy_chunk_size[0], w_noise_index:w_noise_index + self.noisy_chunk_size[1]] = \
            torch.randn_like(half_noisy_crop[:, h_noise_index:h_noise_index + self.noisy_chunk_size[0], w_noise_index:w_noise_index + self.noisy_chunk_size[1]])

        return half_noisy_crop

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        img_crop = self.transform(self.img)
        half_noisy_crop = self._replace_with_noise(img_crop)
        return {'IMG': img_crop, 'CONDITION_IMG': half_noisy_crop}
