import torch
import torch.nn as nn
import torchvision
from pytorch_msssim import SSIM, MS_SSIM, ssim, ms_ssim
import torch.nn.functional as F
from torch.nn import MSELoss as nn_MSELoss
from torch.nn import L1Loss as nn_L1Loss
from . import utils
import lpips


class BaseLoss(torch.nn.Module):
    """Perceptual Loss function based on any torchvision model
    """
    @classmethod
    def from_cfg(cls, cfg):
        return cls()

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        loss = self.loss_fn(x, y)
        return loss

class MSELoss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn_MSELoss()

class L1Loss(BaseLoss):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn_L1Loss()

class PerceptualLoss(torch.nn.Module):
    """Perceptual Loss function based on any torchvision model
    """
    @classmethod
    def from_cfg(cls, cfg):
        base_loss_fn = getattr(nn, cfg.base_loss_class)()
        return cls(cfg.base_loss_network, base_loss_fn, device=cfg.device, resize=cfg.loss_resize)

    def __init__(self, base_loss_network, loss_fn, device="cpu", resize=False):
        """Initialize the loss function

        Args:
            model_name (string): attribute name for torchvision.models
            layer_idxs (list): list of list with indices of layers organized in  blocks
            loss_fn (nn.Loss): loss function to calculate feature loss with
            device (torch.device, optional): device for the model. Defaults to "cpu".
            resize (bool, optional): Whether to resize the input before feature calculations. Defaults to False.
        """
        super().__init__()
        layer_idx_ranges = {
            "vgg19": [3, 8, 17, 26],
            "vgg19_bn": [4, 11, 18, 31, 43]
        }

        layer_idx_range = layer_idx_ranges[base_loss_network]
        features = getattr(torchvision.models, base_loss_network)(pretrained=True).features

        blocks = []
        for i in range(len(layer_idx_range)):
            if i == 0:
                blocks.append(nn.Sequential(features[0:layer_idx_range[0]]))
            else:
                blocks.append(nn.Sequential(features[layer_idx_range[i - 1]:layer_idx_range[i]]))

        self.blocks = nn.ModuleList(blocks)
        self.loss_fn = loss_fn
        self.device = device

        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1), requires_grad=False)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1), requires_grad=False)
        self.resize = resize
        self.to(device)

        for param in self.parameters():
            param.requires_grad = False

        self = self.eval()

    def forward(self, x, y):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        if self.resize:
            x = F.interpolate(x, mode='bilinear', size=(224, 224), align_corners=False)
            y = F.interpolate(y, mode='bilinear', size=(224, 224), align_corners=False)

        x_out = [x]
        y_out = [y]

        for block in self.blocks:
            x_out.append(block(x_out[-1]))
            y_out.append(block(y_out[-1]))

        loss = 0.0
        for i in range(len(x_out)):
            loss = loss + self.loss_fn(x_out[i], y_out[i])
        return loss


class LPIPS(torch.nn.Module):
    """Perceptual Loss function based on any torchvision model
    """
    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg.base_loss_network, device=cfg.device, calibrate=cfg.calibrate)

    def __init__(self, net, device="cpu", calibrate=True, verbose=False):
        super().__init__()
        self.loss_fn = lpips.LPIPS(net=net, lpips=calibrate, verbose=verbose).to(device)
        self.net = net
        self.device = device
        self.calibrate = calibrate

    def forward(self, x, y):
        loss = self.loss_fn(x, y)
        return loss


class NSSIM(nn.Module):
    @classmethod
    def from_cfg(cls, cfg):
        return cls(cfg.nssim_multi_scale, channel=cfg.nssim_channel, channels_as_batch=cfg.nssim_channels_as_batch)

    def __init__(self, multi_scale=True, data_range=1.0, size_average=True, channel=3, channels_as_batch=True):
        """ class for ssim
        Args:
            data_range (float or int, optional): value range of inp images. (usually 1.0 or 255)
            size_average (bool, optional): if size_average=True, ssim of all images will be averaged as a scalar
            channel (int, optional): inp channels (default: 3)
        """

        super().__init__()
        if channels_as_batch:
            if multi_scale:
                self.ssim = MS_SSIM(data_range=data_range, size_average=size_average, channel=1)
            else:
                self.ssim = SSIM(data_range=data_range, size_average=size_average, channel=1)
        else:
            if multi_scale:
                self.ssim = MS_SSIM(data_range=data_range, size_average=size_average, channel=channel)
            else:
                self.ssim = SSIM(data_range=data_range, size_average=size_average, channel=channel)

        self.channels_as_batch = channels_as_batch
        self.channel = channel

    def forward(self, x, y):
        if self.channels_as_batch:
            B, C, H, W = x.shape
            x = x.reshape(B * C, 1, H, W)
            y = y.reshape(B * C, 1, H, W)
        return 1 - self.ssim(x, y)


class TVLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, channel_last=True):
        if channel_last:
            tv_h = torch.abs(x[..., 1:, :, :] - x[..., :-1, :, :]).mean()
            tv_w = torch.abs(x[..., :, 1:, :] - x[..., :, :-1, :]).mean()
        else:
            tv_h = torch.abs(x[..., 1:, :] - x[..., :-1, :]).mean()
            tv_w = torch.abs(x[..., :, 1:] - x[..., :, :-1]).mean()

        loss = tv_h + tv_w

        return loss
