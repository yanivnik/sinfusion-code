import random

from torch.utils.data import Dataset
from torchvision import transforms


class CropSet(Dataset):
    """
    A dataset comprised of crops of a single image or several images.
    """
    def __init__(self, image, crop_size, use_flip=True, dataset_size=5000):
        """
        Args:
            image (torch.tensor): The image to generate crops from.
                                  Can be of shape (C,H,W) or (B,C,H,W) in case of several images.
            crop_size (tuple(int, int)): The spatial dimensions of the crops to be taken.
            use_flip (bool):    Wheather to use horizontal flips of the image.
            dataset_size (int): The amount of images in a single epoch of training. For training datasets,
                                this should be a high number to avoid overhead from pytorch_lightning.
        """
        self.crop_size = crop_size
        self.dataset_size = dataset_size

        transform_list = [transforms.RandomHorizontalFlip()] if use_flip else []
        transform_list += [
            transforms.RandomCrop(self.crop_size, pad_if_needed=False, padding_mode='constant'),
            transforms.Lambda(lambda img: (img[:3, ] * 2) - 1)
        ]

        self.transform = transforms.Compose(transform_list)
        self.img = image

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        # If the training is multi-image, choose one of them to get the crop from
        img = self.img if len(self.img.shape) == 3 else random.choice(self.img)
        img_crop = self.transform(img)
        return {'IMG': img_crop}
