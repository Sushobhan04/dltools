import torch
from torch import tensor
import torch.nn as nn
import copy
from einops import rearrange, repeat


EPS = 1e-8

def homo_to_euclid(points):
    return points[..., :-1] / (points[..., -1:] + EPS)

def euclid_to_homo(points):
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


class PerspectiveCamera(nn.Module):
    @classmethod
    def from_tensors(cls, tensors):
        return cls(tensors[..., 0], tensors[..., 1])

    @classmethod
    def stack(cls, cameras, dim=0):
        intrinsics = torch.stack([camera.intrinsics for camera in cameras], dim=dim)
        extrinsics = torch.stack([camera.extrinsics for camera in cameras], dim=dim)

        return cls(intrinsics, extrinsics)

    def __init__(self, intrinsics, extrinsics=None, device="cpu"):
        super().__init__()
        self.register_buffer("intrinsics", intrinsics.to(device))  # (*, 4, 4)
        self.register_buffer("extrinsics", extrinsics.to(device) if extrinsics is not None else torch.eye(4, device=device).expand_as(intrinsics))  # (*, 4, 4)
        self.device = device
        self.derived_matrices()

    def derived_matrices(self):
        self.K = self.intrinsics[..., :3, :3]
        self.R = self.extrinsics[..., :3, :3]
        self.T = self.extrinsics[..., :3, 3:]
        self.fx = self.K[..., 0, 0]
        self.fy = self.K[..., 1, 1]
        self.cx = self.K[..., 0, 2]
        self.cy = self.K[..., 1, 2]
        self.f = torch.stack([self.fx, self.fy], dim=-1)
        self.c = torch.stack([self.cx, self.cy], dim=-1)

    def clone(self):
        return copy.deepcopy(self)

    def tensors(self):
        return torch.stack([self.intrinsics, self.extrinsics], dim=-1)

    def to(self, device):
        self.intrinsics = self.intrinsics.to(device)
        self.extrinsics = self.extrinsics.to(device)
        self.device = device
        self.derived_matrices()

        return self

    def relative_extrinsics(self, camera):
        return torch.matmul(camera.extrinsics, self.extrinsics.inverse())

    def expand_as(self, pattern, **kwargs):
        tensors = self.tensors()
        tensors = repeat(tensors, pattern, **kwargs)
        return PerspectiveCamera.from_tensors(tensors)

    def unsqueeze(self, dim):
        tensors = self.tensors().unsqueeze(dim)
        return PerspectiveCamera.from_tensors(tensors)

    def translate(self, t):
        dest_extrinsics = self.extrinsics.clone()
        dest_extrinsics[..., :2, 3] = dest_extrinsics[..., :2, 3] + t
        return PerspectiveCamera(self.intrinsics.clone(), dest_extrinsics)

    def relative_distance(self, camera, xy=True):
        if xy:
            out = torch.sqrt(torch.sum(self.relative_distance_vector(camera)[..., :2]**2, dim=-1))
        else:
            out = torch.sqrt(torch.sum(self.relative_distance_vector(camera)**2, dim=-1))

        return out

    def relative_distance_vector(self, camera):
        rel_ext = self.relative_extrinsics(camera)
        T = rel_ext[..., :3, 3]  # (..., 3)

        return T
