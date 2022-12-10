import numpy as np
import torch 
import random

from typing import List, Dict, Optional
from collections import OrderedDict

from .base_carla_policy import BaseCarlaPolicy
from ding.torch_utils.data_helper import to_ndarray
from core.data import CILRSDataset

from torchvision import transforms
from torchvision.models import resnet34
from PIL import Image

class ZeroShotPolicy(BaseCarlaPolicy):
    config = dict()

    def __init__(self, 
                 cfg: dict, 
                 cilrs_path: str='/home/qhzhang/data/cilrs', 
                 sample_len: int=1000, 
                 weight_path: str='/home/qhzhang/data/ckpt/taco.pth.tar'
                 ) -> None:
        super().__init__(cfg, enable_field=set(['eval']))
        self.cilrs_dataset = CILRSDataset(root_dir=cilrs_path)
        self.sample_len = sample_len

        img_list = [[], [], []]
        throttle_list = [[], [], []]
        steering_list = [[], [], []]
        brake_list = [[], [], []]
        
        lst = list(range(len(self.cilrs_dataset)))
        random.shuffle(lst)
        
        i=0
        while len(img_list[0]) < sample_len or len(img_list[1]) < sample_len or len(img_list[2]) < sample_len:
            data = self.cilrs_dataset[lst[i]]
            i+=1
            command = int(data['command'])
            if command == 1:
                k = 0
            elif command == 2:
                k = 1
            else:
                k = 2

            img_list[k].append(data['rgb'].unsqueeze(0))
            throttle_list[k].append(data['throttle'])
            steering_list[k].append(data['steer'])
            brake_list[k].append(data['brake'])

        self.img_vector = [torch.cat(l[:sample_len]) for l in img_list]
        self.throttle_vector = [torch.cat(l[:sample_len]) for l in throttle_list]
        self.steering_vector = [torch.cat(l[:sample_len]) for l in steering_list]
        self.brake_vector = [torch.cat(l[:sample_len]) for l in brake_list]
        self.transform = transforms.Compose([
            transforms.Resize([180, 320]),
            transforms.ToTensor(),
            ])
        
        print('loading network')
        weights = torch.load(weight_path)
        self.net = resnet34(pretrained=False)
        ckpt_dict = OrderedDict()
        for k,v in weights['state_dict'].items():
            if 'module.encoder_q' in k:
                ckpt_dict[k[17:]] = v
        del ckpt_dict['fc.0.weight']
        del ckpt_dict['fc.0.bias']
        del ckpt_dict['fc.2.weight']
        del ckpt_dict['fc.2.bias']

        # self.net.load_state_dict(ckpt_dict, strict=False)
        self.net = torch.nn.Sequential(*(list(self.net.children())[:-1]))
        self.net.eval()
        self.img_feat = [self.net(v).cuda().squeeze() for v in self.img_vector]
        self.net.cuda()

    def _reset(self, data_id:int, noise: bool = False) -> None:
        pass

    def _forward(self, data_id: int, obs: Dict) -> Dict:
        rgb = self.transform(Image.fromarray(obs['rgb']))
        speed = obs['speed']
        command = int(obs['command'])
        if command == 1:
            k=0
        elif command == 2:
            k=1
        else:
            k=2

        obs_rgb_vector = self.net(rgb.unsqueeze(0).cuda()).squeeze().squeeze()
        
        distance = torch.norm(self.img_feat[k] - obs_rgb_vector.unsqueeze(0), p=2, dim=1)
        indices = torch.topk(-1*distance, int(0.1*self.sample_len)).indices

        control = dict()
    
        control['steer'] = float(torch.median(self.steering_vector[k][indices])) * 2 - 1
        # control['brake'] = float(self.brake_vector[indices].mean())
        control['brake'] = 0
        control['throttle'] = float(self.throttle_vector[k][indices].mean())
        print(control)
        return control

    def _forward_eval(self, data: Dict) -> Dict:
        data = to_ndarray(data)
        actions = dict()
        for i in data.keys():
            obs = data[i]
            action = self._forward(i, obs)
            actions[i] = {'action': action}
        return actions

    
if __name__ == '__main__':
    cfg = dict()
    p = ZeroShotPolicy(cfg)

