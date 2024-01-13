import os
from typing import Tuple

import pandas as pd
import torchvision as tv
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image


class ChallengeDataset(Dataset):
    
    def __init__(self, data: pd.DataFrame, mode: str):
        self.data = data
        self.mode = mode

        # Precomputed on training set
        train_mean = [0.478, 0.483, 0.473]
        train_std = [0.214, 0.216, 0.226]

        # Data Augmentation
        self.transform_train = tv.transforms.Compose([
            tv.transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            tv.transforms.RandomApply([tv.transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)], p=0.5),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])
        
        self.transform_val = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(train_mean, train_std)
        ])

    
    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = os.path.join(os.environ["HUK_CHAL"], "data", "imgs", self.data.at[index, "filename"])
        image = Image.open(img_path)

        label_tensor= torch.tensor([self.data.at[index, "perspective_score_hood"], self.data.at[index, "perspective_score_backdoor_left"]])
        image_tensor = self.transform_train(image) if self.mode == "train" else self.transform_val(image)
        
        return image_tensor, label_tensor.float()