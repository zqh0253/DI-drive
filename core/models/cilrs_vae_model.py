import importlib

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from .beta_vae import BetaVAE

class CILRSVAEModel(nn.Module):

    def __init__(
        self,
        backbone='resnet18',
        pretrained=True,
        normalize=True,
        num_branch=6,
        speed_dim=1,
        embedding_dim=512,
        speed_latent_dim=128,
        hidden_size=256,
        input_speed=True,
        predict_speed=True,
        pretrain_path=None,
        bn=False
    ):
        super().__init__()
        self._normalize = normalize
        self._backbone = BetaVAE(in_channels=3, latent_dim=512, )

        if pretrain_path is not None:
            state_dict = torch.load(pretrain_path)
            from collections import OrderedDict

            dct = OrderedDict()

            for k,v in state_dict['state_dict'].items():
                dct[k[6:]] = v
            self._backbone.load_state_dict(dct)

            print(f'SUC: load ckpt from {pretrain_path}')

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
        embedding = 0
        for x in input_images:
            if self._normalize:
                x = self._normalize_imagenet(x)
            embedding += self._backbone.encode(x)[0]
        return embedding

    def forward(self, embedding, speed, command):
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
