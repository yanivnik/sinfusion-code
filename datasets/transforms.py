import random
from common_utils.resize_right import resize


class RandomScaleResize(object):
    def __init__(self, min_scale=0.7, max_scale=1.0):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, x):
        if random.random() < 0.2:  # TODO CONVERT TO CONSTANT IN CONFIG
            return x
        scale = self.min_scale + (self.max_scale - self.min_scale) * random.random()
        return resize(x, scale_factors=scale)
