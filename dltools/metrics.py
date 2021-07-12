import torch
import numpy as np
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import lpips as lpips_lib

def torch_to_numpy(tensor, channel_last=False):
    if channel_last:
        return tensor.cpu().numpy()
    else:
        if tensor.ndim == 3:
            return tensor.cpu().permute(0, 2, 1).numpy()
        elif tensor.ndim == 4:
            return tensor.cpu().permute(0, 2, 3, 1).numpy()

def uint8(tensor):
    return (tensor * 255.0).astype(np.uint8)


def mse(x, y, to_uint8=False, is_torch=True, channel_last=False):
    assert x.shape[0] == y.shape[0]
    if is_torch:
        x = torch_to_numpy(x, channel_last)
        y = torch_to_numpy(y, channel_last)
    if to_uint8:
        x = uint8(x)
        y = uint8(y)

    metric_arr = []
    for i in range(x.shape[0]):
        metric_arr.append(compare_mse(y[i], x[i]))
    metric_mean = np.mean(metric_arr)

    return metric_mean


def psnr(x, y, to_uint8=False, is_torch=True, channel_last=False):
    assert x.shape[0] == y.shape[0]
    if is_torch:
        x = torch_to_numpy(x, channel_last)
        y = torch_to_numpy(y, channel_last)
    if to_uint8:
        x = uint8(x)
        y = uint8(y)

    metric_arr = []
    for i in range(x.shape[0]):
        if to_uint8:
            metric_arr.append(compare_psnr(y[i], x[i], data_range=255))
        else:
            metric_arr.append(compare_psnr(y[i], x[i], data_range=1.0))
    metric_mean = np.mean(metric_arr)

    return metric_mean


def ssim(x, y, to_uint8=True, is_torch=True, gaussian_weights=True, win_size=None, channel_last=False):
    assert x.shape[0] == y.shape[0]
    if is_torch:
        x = torch_to_numpy(x, channel_last)
        y = torch_to_numpy(y, channel_last)
    if to_uint8:
        x = uint8(x)
        y = uint8(y)

    metric_arr = []
    for i in range(x.shape[0]):
        if to_uint8:
            metric_arr.append(compare_ssim(y[i], x[i], gaussian_weights=gaussian_weights, win_size=win_size, multichannel=True, data_range=255))
        else:
            metric_arr.append(compare_ssim(y[i], x[i], gaussian_weights=gaussian_weights, win_size=win_size, multichannel=True, data_range=1.0))
    metric_mean = np.mean(metric_arr)

    return metric_mean


def lpips(x, y, net="vgg", normalize=True, device="cpu", to_numpy=False):
    assert x.shape[0] == y.shape[0]
    if normalize:
        x = (x - 0.5) * 2.0
        y = (y - 0.5) * 2.0

    metric_function = lpips_lib.LPIPS(net=net).to(device)
    metric = metric_function(x, y)

    if to_numpy:
        metric = metric.detach().cpu().numpy()

    return metric

def class_accuracy(x, y, one_hot=True):
    if one_hot:
        x = x.argmax(dim=1)
    acc = torch.sum(x == y, dtype=torch.float32) / y.shape[0]
    return acc
