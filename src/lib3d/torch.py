import torch
from torch import nn
import math
import numpy as np
from scipy.spatial.transform import Rotation


def load_rotation_transform(axis, degrees):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(axis, degrees, degrees=True).as_matrix()
    return torch.from_numpy(transform).float()


def convert_openCV_to_openGL_torch(openCV_poses):
    openCV_to_openGL_transform = (
        torch.tensor(
            [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
            device=openCV_poses.device,
            dtype=openCV_poses.dtype,
        )
        .unsqueeze(0)
        .repeat(openCV_poses.shape[0], 1, 1)
    )
    return torch.bmm(openCV_to_openGL_transform, openCV_poses[:, :3, :3])


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        emb = x[:, :, :, None] * emb[None, None, None, :]  # WxHx3 to WxHxposEnc_size
        emb = emb.reshape(*x.shape[:2], -1)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
