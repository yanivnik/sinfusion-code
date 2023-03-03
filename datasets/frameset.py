import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FrameSet(Dataset):
    """
    A dataset comprised of crops of frames from a single video, with conditioning on previous frames.
    """
    def __init__(self, frames, crop_size=None, dataset_size=5000):
        """
        Args:
            frames (torch.tensor): A tensor of video frames to generate crops from.
            crop_size (tuple(int, int)): The spatial dimensions of the crops to be taken.
            dataset_size (int): The amount of images in a single epoch of training. For training datasets,
                                this should be a high number to avoid overhead from pytorch_lightning.
        """
        self.crop_size = crop_size or (frames.shape[-2], frames.shape[-1])
        self.dataset_size = dataset_size

        self.transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size, pad_if_needed=False),
            transforms.Lambda(lambda img: (img[:3, ] * 2) - 1)
        ])

        self.frames = frames
        self.counter = 0

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        """
        Returns a crop from a random frame, a crop from a previous frame to be used as conditioning, and the frame
        index difference between the two frames.
        All crops are taken from the same spatial coordinates.
        """
        self.counter += 1

        if self.counter < 75000:
            frame_diff = random.choice([1, -1])
        elif self.counter < 150000:
            frame_diff = random.choice([1, -1]) * random.randint(1, 2)
        else:
            frame_diff = random.choice([1, -1]) * random.randint(1, 3)

        if frame_diff > 0:
            frame_idx = random.randrange(frame_diff, self.frames.shape[0])
        else:
            frame_idx = random.randrange(0, self.frames.shape[0] + frame_diff)

        # A hack to make sure both the HR and LR augmentations use the same parameters
        seed = random.randint(1, 10**9)
        random.seed(seed)
        torch.random.manual_seed(seed)
        condition_frame = self.transform(self.frames[frame_idx - frame_diff])

        random.seed(seed)
        torch.random.manual_seed(seed)
        frame = self.transform(self.frames[frame_idx])

        return {
            'CONDITION_IMG': condition_frame,
            'IMG': frame,
            'FRAME': frame_diff}
