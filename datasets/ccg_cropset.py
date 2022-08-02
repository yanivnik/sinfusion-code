from torch.utils.data import Dataset
from torchvision import transforms

from datasets.transforms import RandomScaleResize


class CCGCropSet(Dataset):
    """
    A dataset comprised of crops of various augmentation of a single image,
    used for crop-conditional generation experiments.
    """
    def __init__(self, image, crop_size):
        """
        Args:
            image (torch.tensor): The image to generate crops from.
            crop_size (tuple(int, int)): The spatial dimensions of the crops to be taken.
        """
        self.crop_size = crop_size

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            RandomScaleResize(),
            transforms.RandomCrop(self.crop_size, pad_if_needed=True, padding_mode='constant'),
            transforms.Lambda(lambda img: (img[:3, ] * 2) - 1)
        ])

        self.img = image

    def __len__(self):
        return 5000  # This is a high number to avoid overhead for pytorch_lightning

    def __getitem__(self, item):
        return self.transform(self.img)
