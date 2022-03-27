from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from common_utils.resize_right import resize


class SingleSet(Dataset):
    """
    A Wrapper class for a tiny dataset created from a single image and its augmentations.
    """
    def __init__(self, image_path):
        # TODO ADD TRANSFORMS
        transform = transforms.Compose([transforms.ToTensor()])

        imgs = Image.open(image_path)
        imgs = resize(imgs, (256, 256))
        imgs = transform(imgs)
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)

        L, C, H, W = imgs.shape
        self.dataset_len = L
        self.depth = C
        self.size = H

        self.input_seq = ((imgs * 2.0) - 1.0)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        return self.input_seq[item]
