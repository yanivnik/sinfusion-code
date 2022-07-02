# TODO GIVE CREDIT


# TODO CHECK MORE METRICS FOR BETTER VALIDATION PERFORMANCE

# TODO CONVERT TO METRIC CLASS FOR BETTER CODE STRUCTURE (AFTER CHOOSING BEST METRIC)

import os

import numpy as np
import torch
from scipy import linalg

from common_utils.ben_image import imread
from metrics.inception import InceptionV3  # TODO DELETE AND USE NORMAL torchvision.Inception3 IF POSSIBLE


def get_activations(image, model):
    model.eval()
    pred = model(image)[0]
    pred_arr = pred.cpu().data.numpy().transpose(0, 2, 3, 1).reshape(pred.shape[2] * pred.shape[3], -1)
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(image, model):
    # Code for SIFID
    act = get_activations(image, model)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)

    return mu, sigma


def calculate_raw_patch_statistics(img):
    # Code for PatchFD
    patch_fd = torch.nn.functional.unfold(img, kernel_size=3, stride=1)
    mu = patch_fd.mean(axis=-1).squeeze(0)
    sigma = patch_fd.squeeze(0).cov()
    return mu, sigma


def get_patch_fd_scores(real_image, generated_samples):
    patch_fd_values = np.zeros(len(generated_samples))

    real_m, real_s = calculate_raw_patch_statistics(real_image)
    for i in range(len(generated_samples)):
        fake_m, fake_s = calculate_raw_patch_statistics(generated_samples[i])
        patch_fd_values[i] = calculate_frechet_distance(real_m, real_s, fake_m, fake_s)

    return patch_fd_values


def get_sifid_scores(real_image, generated_samples):
    """
    Calculate the SIFID score of the samples which were generated on a model (which was trained on the real image).
    """
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[64]]).to(device=real_image.device)
    sifid_values = np.zeros(len(generated_samples))

    real_m, real_s = calculate_activation_statistics(real_image, model)
    for i in range(len(generated_samples)):
        fake_m, fake_s = calculate_activation_statistics(generated_samples[i], model)
        sifid_values[i] = calculate_frechet_distance(real_m, real_s, fake_m, fake_s)
    return sifid_values


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    real_image_path = r'..\images\balloons.png'
    fake_images_path = r'fake_images\level4'
    fake_images_paths = os.listdir(fake_images_path)

    real_image = imread(real_image_path)
    fake_images = [imread(os.path.join(fake_images_path, p)) for p in fake_images_paths]

    print("************** SIFID values *****************")
    sifid_values = get_sifid_scores(real_image, fake_images)
    sifid_values_and_paths = sorted(dict(zip(fake_images_paths, sifid_values)).items(), key=lambda kv: kv[1])
    print('\n'.join([f'{s[0]}: {s[1]}' for s in sifid_values_and_paths]))

    # print("************** Patch FD values *****************")
    # patch_fd_scores = get_patch_fd_scores(real_image, fake_images)
    # patch_fd_scores_and_paths = sorted(dict(zip(fake_images_paths, patch_fd_scores)).items(), key=lambda kv: kv[1])
    # print('\n'.join([f'{s[0]}: {s[1]}' for s in patch_fd_scores_and_paths]))

    # TODO CHECK MORE METRICS