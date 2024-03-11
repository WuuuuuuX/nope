import torch.nn as nn
import torch
from src.models.encoder.base_template import (
    BaseFeatureExtractor,
    conv1x1,
    InfoNCE,
    OcclusionAwareSimilarity,
)
from src.models.encoder.resnet import resnet50

import torch.nn.functional as F
import os

os.environ["MPLCONFIGDIR"] = os.getcwd() + "./tmp/"
from src.models.loss import GeodesicError


class FeatureExtractor(BaseFeatureExtractor):
    def __init__(self, descriptor_size, threshold, normalize, **kwargs):
        super(BaseFeatureExtractor, self).__init__()
        self.latent_dim = descriptor_size
        self.normalize = normalize
        self.name = "template"
        self.backbone = resnet50(
            use_avg_pooling_and_fc=False, num_classes=1
        )  # num_classes is useless

        self.projector = nn.Sequential(
            nn.ReLU(inplace=False),
            conv1x1(2048, 256),
            nn.ReLU(inplace=False),
            conv1x1(256, descriptor_size),
        )
        self.encoder = nn.Sequential(self.backbone, self.projector)

        self.metric = GeodesicError()
        self.loss = InfoNCE()
        self.occlusion_sim = OcclusionAwareSimilarity(threshold=threshold)
        self.sim_distance = nn.CosineSimilarity(dim=1)  # eps=1e-2

    @torch.no_grad()
    def encode_image(self, image, mode=None):
        feat = self.backbone(image)
        feat = self.projector(feat)
        if self.normalize:
            feat = F.normalize(feat, dim=1)
        return feat
