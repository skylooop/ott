import gc
import itertools
import multiprocessing
import os

import h5py
from PIL import Image
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from scipy import linalg
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    TensorDataset,
)
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Normalize,
    Pad,
    RandomCrop,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomVerticalFlip,
    Resize,
    ToTensor,
)


def upsample(y, scale_factor=4):
    y = F.interpolate(y, scale_factor = scale_factor, mode='bilinear') # upsample
    return y

def get_loader_stats(loader, inception_fn, inception_params, batch_size=8, n_epochs=1, verbose=False, classes=True):
    pred_arr = []
    
    for epoch in range(n_epochs):
        with torch.no_grad():
            if classes:
                for step, (X, Y) in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                    for i in range(0, len(X), batch_size):
                        start, end = i, min(i + batch_size, len(X))
                        batch = ((X[start:end] + 1) / 2).type(torch.FloatTensor).numpy()
                        batch = batch.transpose(0, 2, 3, 1)
                        pred_arr.append(np.asarray(inception_fn(inception_params, batch)).reshape(end-start, -1))
            else:
                for step, X in enumerate(loader) if not verbose else tqdm(enumerate(loader)):
                    for i in range(0, len(X), batch_size):
                        start, end = i, min(i + batch_size, len(X))
                        batch = ((X[start:end] + 1) / 2).type(torch.FloatTensor).numpy()
                        # pred_arr.append(model(batch)[0].cpu().data.numpy().reshape(end-start, -1))
                        batch = batch.transpose(0, 2, 3, 1)
                        pred_arr.append(np.asarray(inception_fn(inception_params, batch)).reshape(end-start, -1))

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma


def get_pushed_loader_stats(T, loader, inception_fn, inception_params, batch_size=8, n_epochs=1, verbose=False, device='cuda',
                            use_downloaded_weights=False, upgrade=False):
    size = len(loader.dataset)
    pred_arr = []
    mse_arr = []
    from time import time
    
    for epoch in range(n_epochs):
        for step, X in tqdm(enumerate(loader)):
            for i in range(0, len(X), batch_size):
                start, end = i, min(i + batch_size, len(X))
                img_size = X.shape[-1]
                t0 = time()
                inp = X[start:end].numpy().reshape(end - start, 3 * img_size * img_size)
                t1 = time()
                batch = T(
                    inp
                )
                # )[1][-1].x.reshape(end - start, 3, img_size, img_size)
                t2 = time()
                # batch = batch * 0.5 + 0.5
                # batch = batch.transpose(0, 2, 3, 1)
                # pred_arr.append(np.asarray(inception_fn(inception_params, batch)).reshape(end-start, -1))
                t3 = time()
                print(f"{t1 - t0 = }", f"{t2 - t1 = }", f"{t3 - t2 = }")

    pred_arr = np.vstack(pred_arr)
    mu, sigma = np.mean(pred_arr, axis=0), np.cov(pred_arr, rowvar=False)
    gc.collect(); torch.cuda.empty_cache()
    return mu, sigma

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