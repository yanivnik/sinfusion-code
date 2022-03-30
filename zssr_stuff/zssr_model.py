import torch
import torch.nn as nn


class ZSSRNet(nn.Module):
    def __init__(self, depth=8):
        super().__init__()

        self.depth = depth

        # create conv-relu network
        layers = []
        for i in range(self.depth - 1):
            inc, outc = 64, 64
            if i == 0: inc = 3
            layers.append(nn.Conv2d(inc, outc, 3, padding=1, padding_mode='replicate', bias=False))
            layers.append(nn.ReLU())
        # add last layer
        layers.append(nn.Conv2d(64, 3, 3, padding=1, padding_mode='replicate', bias=False))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        inputs = x.clone()
        x = self.body(x)
        x = torch.add(x, inputs)

        return x



#####################
self.hr_father = random_augment(ims=self.hr_fathers_sources,
                                base_scales=[1.0] + self.conf.scale_factors,
                                leave_as_is_probability=self.conf.augment_leave_as_is_probability,
                                no_interpolate_probability=self.conf.augment_no_interpolate_probability,
                                min_scale=self.conf.augment_min_scale,
                                # max_scale=self.conf.augment_min_scale,  # ([1.0] + self.conf.scale_factors)[len(self.hr_fathers_sources)-1],  TODO: change back!
                                max_scale=([1.0] + self.conf.scale_factors)[len(self.hr_fathers_sources) - 1],
                                allow_rotation=self.conf.augment_allow_rotation,
                                scale_diff_sigma=self.conf.augment_scale_diff_sigma,
                                shear_sigma=self.conf.augment_shear_sigma,
                                crop_size=self.conf.crop_size)
