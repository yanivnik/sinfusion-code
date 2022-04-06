from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CropSet(Dataset):
    """
    A dataset comprised of crops of various augmentation of a single image.
    """
    def __init__(self, image_path, crop_size=(100, 100)):
        self.crop_size = crop_size

        # TODO: Think about fine tuning these transformations.
        #   Maybe a rotation that happens all the time isn't a good idea (learn from zssr.random_augment).
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((0, 180), transforms.InterpolationMode.BICUBIC),
            # transforms.RandomResizedCrop TODO DO THE RESIZE WITH resize_right (PROBABLY BETTER)
            transforms.RandomCrop(self.crop_size, pad_if_needed=True, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Lambda(lambda img: (img * 2) - 1)
        ])

        self.img = Image.open(image_path)
        # self.img = transforms.ToTensor()(self.img)
        # self.img = ((self.img * 2.0) - 1.0)

        c, h, w = self.img.shape
        # self.dataset_len = L
        self.depth = c
        self.size = h

    def __len__(self):
        return 1  # TODO CHANGE THIS?

    def __getitem__(self, item):
        return self.transform(self.img)
