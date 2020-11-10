import torch
import numpy as np
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips as lpips_lib
import kornia

def torch_to_numpy(tensor, to_uint8=False):
    image = kornia.tensor_to_image(tensor)
    if to_uint8:
        image = (image * 255.0).astype(np.uint8)

    return image


def mse(x, y, to_uint8=False):
    assert x.shape[0] == y.shape[0]
    x = torch_to_numpy(x, to_uint8)
    y = torch_to_numpy(y, to_uint8)

    metric_arr = []
    for i in range(x.shape[0]):
        metric_arr.append(compare_mse(y[i], x[i]))
    metric_mean = np.mean(metric_arr)

    return metric_mean


def psnr(x, y, to_uint8=False):
    assert x.shape[0] == y.shape[0]
    x = torch_to_numpy(x, to_uint8)
    y = torch_to_numpy(y, to_uint8)

    metric_arr = []
    for i in range(x.shape[0]):
        if to_uint8:
            metric_arr.append(compare_psnr(y[i], x[i], data_range=255))
        else:
            metric_arr.append(compare_psnr(y[i], x[i], data_range=1.0))
    metric_mean = np.mean(metric_arr)

    return metric_mean


def ssim(x, y, to_uint8=True):
    assert x.shape[0] == y.shape[0]
    x = torch_to_numpy(x, to_uint8)
    y = torch_to_numpy(y, to_uint8)

    metric_arr = []
    for i in range(x.shape[0]):
        if to_uint8:
            metric_arr.append(compare_ssim(y[i], x[i], gaussian_weights=True, multichannel=True, data_range=255))
        else:
            metric_arr.append(compare_ssim(y[i], x[i], gaussian_weights=True, multichannel=True, data_range=1.0))
    metric_mean = np.mean(metric_arr)

    return metric_mean


def lpips(x, y, net="vgg", normalize=True, loss_fn=None, device="cpu"):
    assert x.shape[0] == y.shape[0]
    if normalize:
        x = (x - 0.5) * 2.0
        y = (y - 0.5) * 2.0

    if loss_fn is None:
        loss_fn = lpips_lib.LPIPS(net=net).to(device)
    metric = loss_fn(x, y).detach().cpu().numpy()
    metric_mean = np.mean(metric)

    return metric_mean
