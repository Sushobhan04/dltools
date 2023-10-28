import torch

EPS = 1e-8


def homo_to_euclid(points):
    return points[..., :-1] / (points[..., -1:] + EPS)


def euclid_to_homo(points):
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def chain_mm(matrices):
    out = matrices[0]
    for i in range(1, len(matrices)):
        # out = torch.matmul(out, matrices[i])
        out = torch.einsum("...ij,...jk->...ik", out, matrices[i])
    return out


def transform_points(transform, points, euclid=True):
    if isinstance(transform, list):
        transform = chain_mm(transform)

    if euclid:
        assert transform.shape[-1] == points.shape[-1] + 1
        points = euclid_to_homo(points)

    points = torch.einsum("...ij,...j->...i", transform, points)

    if euclid:
        points = homo_to_euclid(points)

    return points
