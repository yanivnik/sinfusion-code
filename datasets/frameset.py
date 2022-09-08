import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class FrameSet(Dataset):
    def __init__(self, frames, crop_size=None, dataset_size=5000):
        self.crop_size = crop_size or (frames.shape[-2], frames.shape[-1])
        self.dataset_size = dataset_size

        self.transform = transforms.Compose([
            transforms.RandomCrop(self.crop_size, pad_if_needed=False),
            transforms.Lambda(lambda img: (img[:3, ] * 2) - 1)
        ])

        self.frames = torch.cat([torch.randn_like(frames[0]).unsqueeze(0), frames])  # First frame isn't conditioned on a previous frame

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        frame_idx = random.randrange(1, self.frames.shape[0])

        # A hack to make sure both the HR and LR augmentations use the same parameters
        seed = random.randint(1, 10**9)
        random.seed(seed)
        torch.random.manual_seed(seed)
        condition_frame = self.transform(self.frames[frame_idx - 1])

        random.seed(seed)
        torch.random.manual_seed(seed)
        frame = self.transform(self.frames[frame_idx])

        return {'CONDITION_IMG': condition_frame, 'IMG': frame, 'FRAME': frame_idx}
