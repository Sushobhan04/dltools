import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from argparse import Namespace
import torch.functional as F


def z(i, fill=4):
    if isinstance(i, int):
        i = str(i)

    i = i.zfill(fill)

    return i

def torch_to_numpy(tensor, channel_last=False):
    if channel_last:
        return tensor.cpu().numpy()
    else:
        if tensor.ndim == 3:
            return tensor.cpu().permute(0, 2, 1).numpy()
        elif tensor.ndim == 4:
            return tensor.cpu().permute(0, 2, 3, 1).numpy()

def regular_meshgrid(shape, normalize=False, device="cpu"):
    grid = torch.meshgrid(*[torch.arange(i, device=device) for i in shape])
    grid = torch.stack(grid, dim=-1)

    if normalize:
        grid = grid / (torch.tensor(shape, device=device) - 1.0)

    return grid

def linspace_meshgrid(shape, device="cpu"):
    grid = torch.meshgrid(*[torch.linspace(-1.0, 1.0, i, device=device) for i in shape])
    grid = torch.stack(grid, dim=-1)
    return grid


def ravel_multi_index(indices, shape):
    assert indices.shape[-1] == len(shape), f"indices.shape[-1] and len(shape) must be equal. Found {indices.shape[-1]} and {len(shape)}"

    out = torch.zeros((indices.shape[:-1]), dtype=indices.dtype, device=indices.device)
    cum_prod = torch.ones((1,), dtype=torch.int64, device=indices.device)
    for i in range(len(shape) - 1, -1, -1):
        out += indices[..., i] * cum_prod
        cum_prod = cum_prod * int(shape[i])

    return out


def update_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def hist_match_grey(source, template, to_int=True):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image
    Arguments:
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    output = interp_t_values[bin_idx].reshape(oldshape)

    if to_int:
        output = output.astype(np.uint8)

    return output


def hist_match(source, template, channel_dim=0, to_int=True):
    equalized_img = []

    for channel in range(source.shape[channel_dim]):
        if channel_dim == 0:
            equalized_img.append(hist_match_grey(source[channel], template[channel], to_int=to_int))
        elif channel_dim == 2:
            equalized_img.append(hist_match_grey(source[:, :, channel], template[:, :, channel], to_int=to_int))
        else:
            print("channel dimension not proper !! ")
            return

    equalized_img = np.array(equalized_img)

    if channel_dim == 2:
        equalized_img = equalized_img.transpose(1, 2, 0)

    return equalized_img

def matmul(a, b):
    shape_a = a.shape
    shape_b = b.shape

    if a.ndim < b.ndim:
        for _ in range(b.ndim - a.ndim):
            a.unsqueeze_(0)
    else:
        for _ in range(a.ndim - b.ndim):
            b.unsqueeze_(0)

    expanded_shape = [max([a.shape[i], b.shape[i]]) for i in range(a.ndim - 2)] + [-1, -1]

    a = a.expand(expanded_shape).reshape(-1, a.shape[-2], a.shape[-1])
    b = b.expand(expanded_shape).reshape(-1, b.shape[-2], b.shape[-1])

    out = torch.bmm(a, b).reshape(expanded_shape[:-2] + [shape_a[-2]] + [shape_b[-1]])
    return out

def chain_mm(matrics):
    out = matrics[0]
    for i in range(1, len(matrics)):
        out = matmul(out, matrics[i])
    return out

def move_to_device(data, device):
    if torch.is_tensor(data):
        data = data.to(device)
    else:
        for key in data.keys():
            if torch.is_tensor(data[key]) and data[key].device != device:
                data[key] = data[key].to(device)
    return data

def one_hot(tensor, size):
    v = torch.zeros((*tensor.shape, size), device=tensor.device)
    range_idx = torch.arange(tensor.shape[0], device=tensor.device)
    v[range_idx, tensor] = 1.0

    return v

def translate_image(self, image, translation):
    B, C, H, W = image.shape

    grid = linspace_meshgrid([H, W], device=image.device)  # (B, H, W, 2)
    grid = grid - translation[:, None, None]  # (B, H, W, 2)

    new_image = F.grid_sample(image, grid, align_corners=True)  # (B, C, H, W)
    return new_image

def circular_translation(r, V):
    theta = torch.arange(V) * 2 * np.pi / V   # (V)
    r_vector = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1) * r  # (V, 2)
    return r_vector

def video_from_frames(filename, frames, fps, codec, verbose=False):
    B, C, H, W = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(filename, fourcc, fps, (W, H))

    frames = torch_to_numpy(frames)[..., ::-1]

    for i, frame in enumerate(frames):
        if verbose:
            print(f"frame {i}")

        video.write(frame)

    video.release()

    if verbose:
        print(f"file saved to {filename}")

def video_from_PIL(filename, frames, fps, codec):
    W, H = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*codec)
    video = cv2.VideoWriter(filename, fourcc, fps, (W, H))

    for frame in frames:
        f = np.asarray(frame)[..., ::-1]
        video.write(f)

    video.release()
