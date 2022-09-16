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

        self.frames = frames
        self.counter = 0

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        self.counter += 1

        if self.counter < 100_000:
            frame_diff = 1
        elif self.counter < 200_000:
            frame_diff = random.randint(1, 2)
        else:
            frame_diff = random.randint(1, 3)

        if frame_diff >= 0:
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

        return {'CONDITION_IMG': condition_frame,
                'IMG': frame,
                'FRAME': frame_diff}

    # TODO REMOVE SOMETIME
    # This is the implementation for the FrameSet.__getitem__ used for simple frame embedding training (not frame diff)
    # def __getitem__(self, item):
    #     frame_idx = random.randrange(1, self.frames.shape[0])
    #
    #     # A hack to make sure both the HR and LR augmentations use the same parameters
    #     seed = random.randint(1, 10 ** 9)
    #     random.seed(seed)
    #     torch.random.manual_seed(seed)
    #     condition_frame = self.transform(self.frames[frame_idx - 1])
    #
    #     random.seed(seed)
    #     torch.random.manual_seed(seed)
    #     frame = self.transform(self.frames[frame_idx])
    #
    #     return {'CONDITION_IMG': condition_frame,
    #             'IMG': frame,
    #             'FRAME': frame_idx}
