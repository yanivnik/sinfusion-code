import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TemporalInterpolationFrameSet(Dataset):
    """
    A dataset comprised of crops of frames from a single video, with conditioning on previous and next frames.
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

        if len(frames) % 2 == 0:
            end_frame = -1
        else:
            end_frame = len(frames)
        self.training_frames = frames[0:end_frame:2]
        self.gt_frames = frames[1:end_frame:2]

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        """
        Returns a crop from a random frame and two concatenated crops (from a previous frame and a next frame) to be
        used as conditioning.
        All crops are taken from the same spatial coordinates.
        """
        frame_idx = random.randrange(0, self.gt_frames.shape[0])

        # A hack to make sure both the HR and LR augmentations use the same parameters
        seed = random.randint(1, 10**9)
        random.seed(seed)
        torch.random.manual_seed(seed)
        condition_frame_1 = self.transform(self.training_frames[frame_idx])

        random.seed(seed)
        torch.random.manual_seed(seed)
        condition_frame_2 = self.transform(self.training_frames[frame_idx + 1])

        condition_frames = torch.cat([condition_frame_1, condition_frame_2], dim=0)
        assert len(condition_frames.shape) == 3 and condition_frames.shape[0] == 6

        random.seed(seed)
        torch.random.manual_seed(seed)
        frame = self.transform(self.gt_frames[frame_idx])

        return {
            'CONDITION_IMG': condition_frames,
            'IMG': frame,
        }
