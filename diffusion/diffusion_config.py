import torch

# TODO IMPLEMENT CONFIG
class DiffusionConfig(object):

    # DEFAULT VALUES
    DEFAULT_TIMESTEPS = 1000
    DEFAULT_IMG_CHANNELS = 3
    DEFAULT_LR = 2e-4
    DEFAULT_OPTIM = torch.optim.Adam
