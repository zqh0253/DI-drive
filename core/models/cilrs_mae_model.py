import importlib

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from . import models_mae

def prepare_mae_model(pretrained, ckpt_dir, arch='mae_vit_large_patch16'):
    model = getattr(models_mae, arch)()
    if pretrained:
        checkpoint = torch.load(ckpt_dir, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
    return model


class CILRSMAEModel(nn.Module):

    def __init__(
        self,
        backbone='resnet18',
        pretrained=True,
        normalize=True,
        num_branch=6,
        speed_dim=1,
        embedding_dim=768,
        speed_latent_dim=128,
        hidden_size=256,
        fix_backbone=False,
        input_speed=True,
        predict_speed=True,
        pretrain_path=None,
        bn=False
    ):
        super().__init__()
        self._normalize = normalize
        self._backbone = prepare_mae_model(pretrained, pretrain_path, 'mae_vit_base_patch16')

        self._num_branch = num_branch
        self._input_speed = input_speed
        self.predict_speed = predict_speed

        # Project input speed measurement to feature size
        if input_speed:
            self._speed_in = nn.Sequential(
                nn.Linear(speed_dim, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, speed_latent_dim),
            )

        # Project feature to speed prediction
        if predict_speed:
            self._speed_out = nn.Sequential(
                nn.Linear(embedding_dim+speed_latent_dim, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(True),
                nn.Linear(hidden_size, speed_dim),
            )

        # Control branches
        fc_branch_list = []
        for i in range(num_branch):
            fc_branch_list.append(
                nn.Sequential(
                    nn.Linear(embedding_dim+speed_latent_dim, hidden_size),
                    nn.ReLU(True),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(True),
                    nn.Linear(hidden_size, 3),
                    nn.Sigmoid(),
                )
            )

        self._branches = nn.ModuleList(fc_branch_list)

        self.bn = bn
        if self.bn:
            self.bn_head = nn.BatchNorm1d(embedding_dim+speed_latent_dim, affine=True)
        self.fix_backbone = fix_backbone
        if self.fix_backbone:
            print('fix backbone')
            self._backbone.eval()
            for p in self._backbone.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.fix_backbone:
            self._backbone.eval()

    def _normalize_imagenet(self, x):
        """
        Normalize input images according to ImageNet standards.
        :Arguments:
            x (tensor): input images
        """
        x = x.clone()
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        return x

    def encode(self, input_images):
        assert NotImplementedError
        return embedding

    def forward(self, input_images, speed, command):
        embedding = 0
        for x in input_images:
            if self._normalize:
                x = self._normalize_imagenet(x)
            emb, _, _= self._backbone.forward_encoder(x, mask_ratio=0)
            emb = emb[:, 0, ...]
            #emb = emb.sum(1)
            embedding += emb

        if len(speed.shape) == 1:
            speed = speed.unsqueeze(1)
        if len(command.shape) == 1:
            command = command.unsqueeze(1)
        if self._input_speed:
            embedding = torch.cat([embedding, self._speed_in(speed)], dim=1)
        if self.bn:
            embedding = self.bn_head(embedding)

        control_pred = 0.
        for i, branch in enumerate(self._branches):
            # Choose control for branch of only active command
            # We check for (command - 1) since navigational command 0 is ignored
            control_pred += branch(embedding) * (i == (command - 1))

        if self.predict_speed:
            speed_pred = self._speed_out(embedding)
            return control_pred, speed_pred

        return control_pred
