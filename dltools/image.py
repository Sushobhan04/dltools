import copy
import os
import pathlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


def resize(img, size=None, scale=None, mode="bilinear"):
    assert len(img.shape) >= 3, "Image must have at least 3 dimensions"

    if size is not None:
        assert len(size) == 2, "Size must be a tuple of 2 integers"
    elif scale is not None:
        size = (int(img.shape[-2] * scale), int(img.shape[-1] * scale))
    else:
        raise ValueError("Either size or scale must be provided")

    if torch.is_tensor(size):
        size = size.tolist()

    with_batch = len(img.shape) == 4

    if not with_batch:
        img = img[None]

    if img.dtype == torch.uint8:
        img = torchvision.transforms.functional.resize(img, size)
    else:
        img = F.interpolate(img, size, mode=mode)

    if not with_batch:
        img = img[0]

    return img


def torch2np(img):
    with_batch = len(img.shape) == 4

    if with_batch:
        img = img.permute(0, 2, 3, 1)
    else:
        img = img.permute(1, 2, 0)

    img = img.detach().cpu().numpy()
    return img


def np2torch(img):
    if torch.is_tensor(img):
        return img

    with_batch = len(img.shape) == 4

    img = torch.from_numpy(img)

    if with_batch:
        img = img.permute(0, 3, 1, 2)
    else:
        img = img.permute(2, 0, 1)[None]

    return img


def cv2torch(img, bgr=False):
    img = img[..., ::-1].copy()
    img = np2torch(img)
    return img


def torch2cv(img):
    img = torch2np(img)
    img = img[..., ::-1].copy()
    return img


def read_image(path, verbose=True):
    if isinstance(path, pathlib.Path):
        path = str(path)

    if not os.path.exists(path):
        if verbose:
            print(f"{path} does not exist")
        return

    img = cv2.imread(path)

    if img is None:
        if verbose:
            print(f"Could not read image from {path}")
        return

    img = img[..., ::-1].copy()
    img = np2torch(img)
    return img


def save_image(img, path):
    assert img.ndim >= 3, "Image must have at least 3 dimensions"

    if img.ndim == 4:
        assert img.shape[0] == 1, "Only batch size 1 is supported"
        img = img[0]

    img = torch2cv(img).copy()
    if img.dtype == np.float32:
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)

    if isinstance(path, str):
        path = pathlib.Path(path)

    path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(path.as_posix(), img)


def float32_to_uint8(img):
    img = img * 255.0
    img = torch.clamp(img, 0.0, 255.0)
    img = img.type(torch.uint8)
    return img


def uint8_to_float32(img):
    img = img.type(torch.float32)
    img = img / 255.0
    return img


def center_pad(img, size):
    H, W = img.shape[-2:]
    h, w = size

    top = (h - H) // 2
    left = (w - W) // 2
    bottom = h - H - top
    right = w - W - left

    img = F.pad(img, (left, right, top, bottom))

    return img, [top, left, bottom, right]


def ar_preserving_resize(img, size, mode=None, return_transform=False):
    H, W = img.shape[-2:]
    h, w = size

    h_ratio = h / H
    w_ratio = w / W

    a_ratio = min(h_ratio, w_ratio)
    min_size = (int((a_ratio * H) // 1), int((a_ratio * W) // 1))

    out = resize(img, size=min_size, mode=mode)
    out, pad = center_pad(out, size)

    if return_transform:
        transform = torch.eye(3, device=img.device, dtype=torch.float32)
        transform[:2, :2] = transform[:2, :2] * a_ratio
        transform[0, 2] = pad[0]
        transform[1, 2] = pad[1]
        return out, transform

    return out


def ar_preserving_bbox(bbox, target_ratio, expand=True):
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    h, w = bbox[2] - bbox[0], bbox[3] - bbox[1]

    ratio = h / w

    if ratio == target_ratio:
        return bbox

    update_height = (ratio < target_ratio and expand) or (
        ratio > target_ratio and not expand
    )  # (whether to update width or height)

    if update_height:
        h = w * target_ratio
    else:
        w = h / target_ratio

    new_bbox = copy.deepcopy(bbox)

    new_bbox[0] = int(cx - h / 2.0)
    new_bbox[1] = int(cy - w / 2.0)
    new_bbox[2] = new_bbox[0] + h
    new_bbox[3] = new_bbox[1] + w

    return new_bbox


def ar_preserving_crop(img, bbox, ar, expand=True, return_transform=False):
    bbox = ar_preserving_bbox(bbox, ar, expand)
    out = crop_tlbr(img, bbox, safety_check=True, return_transform=return_transform)

    return out


def crop_tlbr(img, bbox, fill_mode="zeros", padded=False, return_transform=False):
    if padded:
        B, C, H, W = img.shape
        device = img.device
        dtype = img.dtype
        h, w = bbox[2] - bbox[0], bbox[3] - bbox[1]

        if fill_mode == "zeros":
            out = torch.zeros((B, C, h, w), device=device, dtype=dtype)
        elif fill_mode == "ones":
            out = torch.ones((B, C, h, w), device=device, dtype=dtype)
            if dtype == torch.uint8:
                out = out * 255

        out_start_h, out_start_w = max(0, -bbox[0]), max(0, -bbox[1])
        out_end_h, out_end_w = min(h, max(0, H - bbox[0])), min(w, max(0, W - bbox[1]))

        inp_start_h, inp_start_w = max(0, bbox[0]), max(0, bbox[1])
        inp_end_h, inp_end_w = min(H, max(0, bbox[2])), min(W, max(0, bbox[3]))

        out[..., out_start_h:out_end_h, out_start_w:out_end_w] = img[
            ..., inp_start_h:inp_end_h, inp_start_w:inp_end_w
        ]
    else:
        channel_last = False
        if img.shape[-1] in (3, 1):
            channel_last = True

        if channel_last:
            out = img[..., bbox[0] : bbox[2], bbox[1] : bbox[3], :]
        else:
            out = img[..., bbox[0] : bbox[2], bbox[1] : bbox[3]]

    if return_transform:
        transform = torch.eye(3, device=img.device, dtype=torch.float32)
        transform[0, 2] = -bbox[0]
        transform[1, 2] = -bbox[1]

        return out, transform

    return out


def largest_bbox(bboxes):
    tl = bboxes[:, :2].min(dim=0)
    br = bboxes[:, 2:].max(dim=0)
    bbox = torch.cat([tl, br], dim=-1)

    return bbox


def sqbbox(bboxes, min_size=None):
    sides = torch.maximum(
        bboxes[..., 2] - bboxes[..., 0], bboxes[..., 3] - bboxes[..., 1]
    )
    if min_size is not None:
        sides = torch.maximum(sides, min_size)
    centers = (bboxes[..., :2] + bboxes[..., 2:4]) / 2.0
    tl = centers - (sides // 2).int()
    br = tl + sides

    bbox = torch.cat([tl, br], dim=-1)

    return bbox
