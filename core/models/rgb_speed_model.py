import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union

from ding.torch_utils import MLP
import torchvision.models as models
from collections import OrderedDict

class RGBSpeedConvEncoder(nn.Module):
    """
    Convolutional encoder of Bird-eye View image and speed input. It takes a BeV image and a speed scalar as input.
    The BeV image is encoded by a convolutional encoder, to get a embedding feature which is half size of the
    embedding length. Then the speed value is repeated for half embedding length time, and concated to the above
    feature to get a final feature.

    :Arguments:
        - obs_shape (Tuple): BeV image shape.
        - hidden_dim_list (List): Conv encoder hidden layer dimension list.
        - embedding_size (int): Embedding feature dimensions.
        - kernel_size (List, optional): Conv kernel size for each layer. Defaults to [8, 4, 3].
        - stride (List, optional): Conv stride for each layer. Defaults to [4, 2, 1].
    """

    def __init__(
            self,
            obs_shape: Tuple,
            embedding_size: int,
            task_pretrained: bool,
            img_pretrained: bool,
            fix_perception: bool
    ) -> None:
        super().__init__()

        self._obs_shape = obs_shape
        self._embedding_size = embedding_size
        self._model = models.resnet34(pretrained=img_pretrained)
        # d = torch.load('/home/yhxu/qhzhang/workspace/only_steering.pt')

        # newd = OrderedDict()
        # for k,v in d.items():
        #     if 'module._perc' in k:
        #         newd[k[19:]] = v
        # if task_pretrained:
        #     self._model.load_state_dict(newd)
            
        d = torch.load('/home/qhzhang/data/ckpt/taco_v5.pth.tar')
        newd = OrderedDict()
        for k,v in d['state_dict'].items():
            if 'module.encoder_q' in k:
                newd[k[17:]] = v
        del newd['fc.0.weight']
        del newd['fc.0.bias']
        del newd['fc.2.weight']
        del newd['fc.2.bias']
        if task_pretrained:
            self._model.load_state_dict(newd, strict=False)

        self.fix_perception = fix_perception
        self.perception = torch.nn.Sequential(*(list(self._model.children())[:-1]))
        # self.perception.load_state_dict(newdd)
        if fix_perception:
            self.perception.eval()
            for p in self.perception.parameters():
                p.requires_grad=False
        flatten_size = 512
        self._mid = nn.Linear(flatten_size, self._embedding_size // 2)
        self.ln = nn.LayerNorm(flatten_size)

    def train(self, mode=True):
        super().train(mode)
        if self.fix_perception:
            self.perception.eval()

    def _get_flatten_size(self) -> int:
        test_data = torch.randn(1, *self._obs_shape)
        with torch.no_grad():
            output = self._model(test_data)
        return output.shape[1]

    def forward(self, data: Dict) -> torch.Tensor:
        """
        Forward computation of encoder

        :Arguments:
            - data (Dict): Input data, must contain 'birdview' and 'speed'

        :Returns:
            torch.Tensor: Embedding feature.
        """
        image = data['rgb'].permute(0, 3, 1, 2)
        speed = data['speed']
        x = self.perception(image).squeeze(-1).squeeze(-1)
        # x = torch.nn.functional.normalize(x, dim=1)
        x = self.ln(x)
        x = self._mid(x)
        speed_embedding_size = self._embedding_size - self._embedding_size // 2
        speed_vec = torch.unsqueeze(speed, 1).repeat(1, speed_embedding_size)
        h = torch.cat((x, speed_vec), dim=1)
        return h
