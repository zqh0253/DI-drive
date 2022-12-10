import os
import numpy as np
from typing import Any, Dict
import torch
from torch.utils.data import Dataset

from PIL import Image
from core.utils.others.image_helper import read_image


class CILRSDataset(Dataset):

    def __init__(self, root_dir: str, transform: bool = False, preloads: str = None, shrink_ratio=1, mae_resize=False) -> None:
        self._root_dir = root_dir
        self._transform = transform

        preload_file = preloads
        if preload_file is not None:
            print('[DATASET] Loading from NPY')
            self._sensor_data_names, self._measurements = np.load(preload_file, allow_pickle=True)
        self.shrink_ratio = shrink_ratio
        self.mae_resize=  mae_resize

    def __len__(self) -> int:
        return len(self._sensor_data_names) // self.shrink_ratio

    def __getitem__(self, index: int) -> Any:
        img_path = os.path.join(self._root_dir, self._sensor_data_names[index])
        
        img = read_image(img_path)
        assert img.shape[2] == 3, f'{img.shape}, {img_path}'
        if self.mae_resize:
            img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
            img = np.array(img)
        if self._transform:
            img = img.transpose(2, 0, 1)
            img = img / 255.
        img = img.astype(np.float32)
        img = torch.from_numpy(img).type(torch.FloatTensor)

        measurements = self._measurements[index].copy()
        data = dict()
        data['rgb'] = img
        for k, v in measurements.items():
            v = torch.from_numpy(np.asanyarray([v])).type(torch.FloatTensor)
            data[k] = v
        return data
