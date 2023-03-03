import random
from common_utils.resize_right import resize


class RandomScaleResize(object):
    def __init__(self, min_scale=0.7, max_scale=1.0, rescale_threshold=0.8):
        """
        Resize the image using a random scale in [min_scale, max_scale].
        The resizing only occurs with probability of rescale_threshold.
        """
        self.rescale_threshold = rescale_threshold
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, x):
        if random.random() < self.rescale_threshold:
            scale_h = self.min_scale + (self.max_scale - self.min_scale) * random.random()
            scale_w = self.min_scale + (self.max_scale - self.min_scale) * random.random()
            return resize(x, scale_factors=(scale_h, scale_w))
        else:
            return x
