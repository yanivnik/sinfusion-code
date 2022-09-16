from torch.utils.data import Dataset
from torchvision import transforms


class CropSet(Dataset):
    """
    A dataset comprised of crops of various augmentation of a single image.
    """
    def __init__(self, image, crop_size, use_flip=True, dataset_size=5000):
        """
        Args:
            image (torch.tensor): The image to generate crops from.
            crop_size (tuple(int, int)): The spatial dimensions of the crops to be taken.
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
        return {'IMG': self.transform(self.img)}
