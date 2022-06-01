import random
from torchvision import transforms


class RandomScaleResize(object):
    def __init__(self, min_scale=0.7, max_scale=1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, x):
        if random.random() < 0.2:  # TODO CONVERT TO CONSTANT IN CONFIG
            return x
        scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
        resized_size = (int(x.size[1] * scale), int(x.size[0] * scale))  # reverse dimension order because of functional.resize logic
        return transforms.functional.resize(x, resized_size)