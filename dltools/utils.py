import importlib
import torch
import numpy as np
import matplotlib.pyplot as plt
import kornia
from PIL import Image


class Config(object):
    def __init__(self):
        pass


def name_to_class(class_name, module_name):
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def z(i, fill=4):
    if isinstance(i, int):
        i = str(i)

    i = i.zfill(fill)

    return i


def regular_meshgrid(shape, device="cpu"):
    grid = torch.meshgrid(*[torch.arange(i) for i in shape])
    grid = torch.stack(grid, dim=-1).to(device)
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


def imshow(img, figsize=10, pytorch=True, colorbar=False, grid=True, cmap=None):
    if pytorch:
        img = img.detach().cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(figsize, figsize))
    plt.imshow(img, cmap=cmap)
    if colorbar:
        plt.colorbar()
    if grid:
        plt.grid()


def plot(x, y, params):
    plt.figure(figsize=params["figsize"])
    plt.plot(x, y, params["marker"], linestyle=params["linestyle"])
    plt.grid(params["grid"])
    plt.title(params["title"])
    plt.xlabel(params["xlabel"])
    plt.ylabel(params["ylabel"])
    plt.xlim(*params["xlim"])
    plt.ylim(*params["ylim"])

    if params["save_file"] is not None:
        plt.savefig(params["save_file"], dpi=params["dpi"])


def merge_dims(tensor, dim):
    out_shape = list(tensor.shape)
    out_shape = out_shape[: dim[0]] + [np.prod(tensor.shape[dim[0]:dim[1] + 1])] + out_shape[dim[1] + 1:]
    return tensor.reshape(out_shape)

def unmerge_dims(tensor, dim, shape):
    out_shape = list(tensor.shape)
    out_shape = out_shape[: dim] + list(shape) + out_shape[dim + 1:]
    return tensor.reshape(out_shape)

def matmul(a, b):
    shape_a = a.shape
    shape_b = b.shape

    if a.ndim < b.ndim:
        for _ in range(b.ndim - a.ndim):
            a = a.unsqueeze(0)
    else:
        for _ in range(a.ndim - b.ndim):
            b = b.unsqueeze(0)

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


def tensor_to_image(tensor, to_uint8=True):
    image = kornia.tensor_to_image(tensor)
    if to_uint8:
        image = (image * 255.0).astype(np.uint8)

    if image.ndim == 4:
        image = image[0]

    image = Image.fromarray(image)
    return image

def move_to_device(data, device):
    if torch.is_tensor(data):
        data = data.to(device)
    else:
        for key in data.keys():
            if torch.is_tensor(data[key]) and data[key].device != device:
                data[key] = data[key].to(device)
    return data
