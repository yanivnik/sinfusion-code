import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from common_utils.resize_right import resize
from datasets.transforms import RandomScaleResize


class SRCropSet(Dataset):
    """
    A dataset comprised of pairs of augmented crops from a (HighRes, LowRes) image pair.
    All augmentations and crops are identical on both images.
    """
    def __init__(self, hr, lr, crop_size):
        """
        Args:
            hr (PIL.Image): The high resolution image to generate crops from.
            lr (PIL.Image): The low resolution image to generate crops from.
            crop_size (tuple(int, int)): The spatial dimensions of the crops to be taken.
        """
        self.crop_size = crop_size

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            RandomScaleResize(),
            transforms.RandomCrop(self.crop_size, pad_if_needed=True, padding_mode='reflect'),
            transforms.Lambda(lambda img: (img[:3, ] * 2) - 1)
        ])

        self.hr = hr
        self.lr = resize(lr, out_shape=self.hr.size[::-1])

    def __len__(self):
        return 5000  # This is a high number to avoid overhead for pytorch_lightning

    def __getitem__(self, item):
        # A quick hack to make sure both the HR and LR augmentations use the same parameters
        seed = random.randint(1, 10**9)
        random.seed(seed)
        torch.random.manual_seed(seed)
        hr_crop = self.transform(self.hr)

        random.seed(seed)
        torch.random.manual_seed(seed)
        lr_crop = self.transform(self.lr)

        #from PIL import Image
        #self.hr.save(r'/home/yanivni/data/tmp/hr.png')
        #self.lr.save(r'/home/yanivni/data/tmp/lr.png')
        #Image.fromarray((((hr_crop + 1) / 2) * 255).moveaxis(0, 2).to(dtype=torch.uint8).cpu().numpy()).save(r'/home/yanivni/data/tmp/hr_crop.png')
        #Image.fromarray((((lr_crop + 1) / 2) * 255).moveaxis(0, 2).to(dtype=torch.uint8).cpu().numpy()).save(r'/home/yanivni/data/tmp/lr_crop.png')
        return {'HR': hr_crop, 'LR': lr_crop}
