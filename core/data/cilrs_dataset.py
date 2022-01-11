# import os
# import numpy as np
# from typing import Any, Dict
# import torch
# from torch.utils.data import Dataset
# 
# from core.utils.others.image_helper import read_image
# 
# 
# class CILRSDataset(Dataset):
# 
#     def __init__(self, root_dir: str, transform: bool = False, preloads: str = None) -> None:
#         self._root_dir = root_dir
#         self._transform = transform
# 
#         preload_file = preloads
#         if preload_file is not None:
#             print('[DATASET] Loading from NPY')
#             self._sensor_data_names, self._measurements = np.load(preload_file, allow_pickle=True)
# 
#     def __len__(self) -> int:
#         return len(self._sensor_data_names)
# 
#     def __getitem__(self, index: int) -> Any:
#         img_path = os.path.join(self._root_dir, self._sensor_data_names[index])
#         img = read_image(img_path)
#         if self._transform:
#             img = img.transpose(2, 0, 1)
#             img = img / 255.
#         img = img.astype(np.float32)
#         img = torch.from_numpy(img).type(torch.FloatTensor)
# 
#         measurements = self._measurements[index].copy()
#         data = dict()
#         data['rgb'] = img
#         for k, v in measurements.items():
#             v = torch.from_numpy(np.asanyarray([v])).type(torch.FloatTensor)
#             data[k] = v
#         return data
import torch
import os
from torch.utils.data import Dataset
from typing import Any, Dict

from torchvision import transforms
import numpy as np
from PIL import Image

class CILRSDataset(Dataset):
    def __init__(self, root_dir: str, npy_name: str = 'cilrs_datasets_train.npy') -> None:
        self.root_dir = root_dir
        self.sensor_name, self.measurement = np.load(os.path.join(self.root_dir, npy_name), allow_pickle=True)
        self.trans = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.sensor_name)

    def __getitem__(self, index: int) -> Any:
        img_path = os.path.join(self.root_dir, 'rgb', self.sensor_name[index])
        img = Image.open(img_path)
        img_np = np.array(img)
        img_torch = self.trans(img_np)

        mea = self.measurement[index].copy()

        data=dict()
        data['rgb'] = img_torch
        for k, v in mea.items():
            v = torch.from_numpy(np.asanyarray([v])).type(torch.FloatTensor)
            data[k] = v
        return data

if __name__ == '__main__':
    dataset = CILRSDataset(root_dir='/home/qhzhang/data/cilrs')
    item = dataset[0]
    print(item)
