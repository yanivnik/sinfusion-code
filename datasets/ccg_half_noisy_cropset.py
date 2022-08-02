import random

import torch
import torchvision.utils
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.transforms import RandomScaleResize


class CCGHalfNoisyCropSet(Dataset):
    """
    A dataset comprised of pairs of augmented crops. Each crop pair includes a normal crop,
    and a second, half noisy crop. In this context "half noisy" means that a specific percentage
    of the crop is complete noise, while the other is a normal image. The two parts are thresholded
    somewhere in the image (i.e. - the noisy part is continuous and the normal part is continuous).

    The half noisy crop is used as a conditioning signal to generate the normal crop, which should match it
    in the normal part.
    used for crop-conditional generation experiments.
    """
    def __init__(self, image, crop_size, gamma=0.5):
        """
        Args:
            image (torch.tensor): The image to generate crops from.
            crop_size (tuple(int, int)): The spatial dimensions of the crops to be taken.
            gamma (float): Which percentage of the "half noisy" crop to use as pure noise.
        """
        self.crop_size = crop_size
        self.gamma = gamma

        self.transform = transforms.Compose([
            # transforms.Pad(padding=(self.crop_size[1] // 4, self.crop_size[0] // 4)),
            transforms.RandomHorizontalFlip(),
            RandomScaleResize(),
            transforms.RandomCrop(self.crop_size, pad_if_needed=True, padding_mode='constant'),
            transforms.Lambda(lambda img: (img[:3, ] * 2) - 1)
        ])

        self.img = image

    def _replace_with_noise(self, crop):
        half_noisy_crop = crop.clone()

        # OLD
        # # The choice threshold is proportional to the number of right, bottom, or bottom-right
        # # conditional crops that will be encountered during sampling
        # choice = random.random()
        # choice_threshold = 1 / ((1 / self.gamma) + 1) # TODO RENAME
        # h_noise_index = int(-self.gamma * self.crop_size[0])
        # w_noise_index = int(-self.gamma * self.crop_size[1])
        # if choice < choice_threshold:
        #     # Replace the bottom gamma-part of the image with noise
        #     half_noisy_crop[:, h_noise_index:, :] = torch.randn_like(half_noisy_crop[:, h_noise_index:, :])
        # elif choice_threshold <= choice < 2 * choice_threshold:
        #     # Replace the right gamma-part of the image with noise
        #     half_noisy_crop[:, :, w_noise_index:] = torch.randn_like(half_noisy_crop[:, :, w_noise_index:])
        # else:
        #     # Replace the bottom right (gamma^2)-corner of the image with noise
        #     half_noisy_crop[:, h_noise_index:, w_noise_index:] = \
        #         torch.randn_like(half_noisy_crop[:, h_noise_index:, w_noise_index:])

        # Change a chunk of the image to be complete noise in a random location
        h_noise_crop_size = int((1 - self.gamma) * self.crop_size[0])
        w_noise_crop_size = int((1 - self.gamma) * self.crop_size[1])
        h_noise_index = random.randint(0, h_noise_crop_size - 1)
        w_noise_index = random.randint(0, w_noise_crop_size - 1)
        half_noisy_crop[:, h_noise_index:h_noise_index + h_noise_crop_size, w_noise_index:w_noise_index + w_noise_crop_size] = \
            torch.randn_like(half_noisy_crop[:, h_noise_index:h_noise_index + h_noise_crop_size, w_noise_index:w_noise_index + w_noise_crop_size])

        return half_noisy_crop

    def __len__(self):
        return 5000  # This is a high number to avoid overhead for pytorch_lightning

    def __getitem__(self, item):
        img_crop = self.transform(self.img)
        half_noisy_crop = self._replace_with_noise(img_crop)
        #return {'IMG': img_crop, 'CONDITION_IMG': half_noisy_crop}
        return {'HR': img_crop, 'LR': half_noisy_crop} # TODO RENAME
