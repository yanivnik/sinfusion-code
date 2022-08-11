# TODO GIVE CREDIT

import os
import torch
import torch.nn.functional as F
from common_utils.ben_image import imread


def duplicate_to_match_lengths(arr1, arr2):
    """
    Duplicates randomly selected entries from the smaller array to match its size to the bigger one
    :param arr1: (r, n) torch tensor
    :param arr2: (r, m) torch tensor
    :return: (r,max(n,m)) torch tensor
    """
    if arr1.shape[1] == arr2.shape[1]:
        return arr1, arr2
    elif arr1.shape[1] < arr2.shape[1]:
        tmp = arr1
        arr1 = arr2
        arr2 = tmp

    b = arr1.shape[1] // arr2.shape[1]
    arr2 = torch.cat([arr2] * b, dim=1)
    if arr1.shape[1] > arr2.shape[1]:
        indices = torch.randperm(arr2.shape[1])[:arr1.shape[1] - arr2.shape[1]]
        arr2 = torch.cat([arr2, arr2[:, indices]], dim=1)

    return arr1, arr2


def get_swd_scores(real_image, generated_samples):
    patch_size = 7
    num_proj = 256

    b, c, h, w = real_image.shape

    # Sample random normalized projections
    rand = torch.randn(num_proj, c * patch_size ** 2).to(real_image.device)
    rand = rand / torch.norm(rand, dim=1, keepdim=True)
    rand = rand.reshape(num_proj, c, patch_size, patch_size)

    # Project patches
    projx = F.conv2d(real_image, rand).transpose(1, 0).reshape(num_proj, -1)
    projy = F.conv2d(generated_samples, rand).transpose(1, 0).reshape(num_proj, -1)

    # Duplicate patches if number does not equal
    projx, projy = duplicate_to_match_lengths(projx, projy)

    # Sort and compute L1 loss
    projx, _ = torch.sort(projx, dim=1)
    projy, _ = torch.sort(projy, dim=1)

    loss = torch.abs(projx - projy).mean()

    return loss


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    real_image_path = r'images/balloons.png'
    fake_images_path = r'metrics/fake_images/level4'
    fake_images_paths = os.listdir(fake_images_path)

    real_image = imread(real_image_path)
    fake_images = [imread(os.path.join(fake_images_path, p)) for p in fake_images_paths]

    print("************** SWD values *****************")
    swd_values = [get_swd_scores(real_image, fake_image) for fake_image in fake_images]
    swd_values_and_paths = sorted(dict(zip(fake_images_paths, swd_values)).items(), key=lambda kv: kv[1])
    print('\n'.join([f'{s[0]}: {s[1]}' for s in swd_values_and_paths]))
