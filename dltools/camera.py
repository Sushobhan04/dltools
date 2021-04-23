import torch
import torch.nn as nn
import copy


EPS = 1e-8

def homo_to_euclid(points):
    return points[..., :-1] / (points[..., -1:] + EPS)

def euclid_to_homo(points):
    points_shape = list(points.shape)
    points_shape[-1] = 1
    return torch.cat([points, torch.ones(points_shape, device=points.device)], dim=-1)


class PerspectiveCamera(nn.Module):
    @classmethod
    def from_tensors(cls, tensors):
        return cls(tensors[..., 0], tensors[..., 1])

    @classmethod
    def collate(cls, cameras):
        intrinsics = torch.cat([camera.intrinsics for camera in cameras], dim=0)
        extrinsics = torch.cat([camera.extrinsics for camera in cameras], dim=0)

        return cls(intrinsics, extrinsics)

    def __init__(self, intrinsics, extrinsics=None):
        super().__init__()
        self.register_buffer("intrinsics", intrinsics)  # (*, 4, 4)
        self.register_buffer("extrinsics", extrinsics if extrinsics is not None else torch.eye(4).expand_as(intrinsics))  # (*, 4, 4)
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

    def relative_extrinsics(self, camera):
        return torch.matmul(camera.extrinsics, torch.inverse(self.extrinsics))

    def expand_dim(self, dim):
        if isinstance(dim, int):
            tensors = self.tensors().unsqueeze(dim)
        elif isinstance(dim, list):
            tensors = self.tensors()
            for d in dim:
                tensors = tensors.unsqueeze(d)
        else:
            raise f"dim must be either int or list. Founf {type(dim)}"

        return PerspectiveCamera.from_tensors(tensors)

    def translate(self, t):
        dest_extrinsics = self.extrinsics.clone()
        dest_extrinsics[..., :2, 3] = dest_extrinsics[..., :2, 3] + t
        return PerspectiveCamera(self.intrinsics.clone(), dest_extrinsics)

    def xy_distance(self, camera, xy=True):
        if xy:
            out = torch.sqrt(torch.sum(self.xyz_distance(camera)[..., :2]**2, dim=-1))
        else:
            out = torch.sqrt(torch.sum(self.xyz_distance(camera)**2, dim=-1))

        return out

    def xyz_distance(self, camera):
        rel_ext = self.relative_extrinsics(camera)
        T = rel_ext[..., :3, 3]  # (..., 3)

        return T
