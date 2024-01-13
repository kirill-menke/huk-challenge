import os
import json
import sys
from typing import Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision as tv
from sklearn.model_selection import train_test_split

from data import ChallengeDataset
from trainer import Trainer


def train_val_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Perform a stratified train-validation-split according to target groups """

    hood_score = df['perspective_score_hood'].to_numpy()
    backdoor_left_score = df['perspective_score_backdoor_left'].to_numpy()

    # Ensure a similar distribution of the scores in the train and validation set
    hood_score = np.digitize(hood_score, np.linspace(0, 1, 10))
    backdoor_left_score = np.digitize(backdoor_left_score, np.linspace(0, 1, 10))
    scores = list(zip(hood_score, backdoor_left_score))

    # Mask out samples that are unique in their group
    unique_samples = set(group for group, val in Counter(scores).items() if val == 1)
    mask = np.array([sample not in unique_samples for sample in scores])
    
    train, val = train_test_split(df[mask], test_size=0.2, shuffle=True, stratify=np.array(scores)[mask])
    
    # Add the unique samples to the validation set
    val = pd.concat([val, df[~mask]])

    val.reset_index(inplace=True)
    train.reset_index(inplace=True)

    return train, val


def train(config: dict) -> None:
    """ Train a ResNet model with the given hyperparameters """

    # Load the data and split into training and validation set
    df_path = os.path.join(os.environ["HUK_CHAL"], "data", "car_imgs_4000.csv")
    df = pd.read_csv(df_path, sep=',')
    train, val = train_val_split(df)

    # Create data loaders for training and validation set
    train_dl = torch.utils.data.DataLoader(ChallengeDataset(train, "train"), batch_size=config["batch_size"], num_workers=4, shuffle=True)
    val_dl = torch.utils.data.DataLoader(ChallengeDataset(val, "val"), batch_size=config["batch_size"], num_workers=4, shuffle=True)

    # Create an instance of a pretrained ResNet model
    res_net = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V1)
    res_net.fc = nn.Sequential(nn.Linear(res_net.fc.in_features, 2), nn.Sigmoid())

    # Create an optimizer and a loss function
    optim = torch.optim.SGD(res_net.parameters(), lr=config["lr"], momentum=config["momentum"])
    # weight_decay=config["weight_decay"]
    loss = nn.MSELoss()

    # Learning rate decay
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[10, 15], gamma=0.1)

    # Start training
    trainer = Trainer(res_net, loss, optim, train_dl, val_dl, scheduler, enable_log=config["enable_log"], checkpoint=config["checkpoint"], cuda=config["cuda"])
    trainer.fit(epochs=config["epochs"])


if __name__ == "__main__":
    with open(sys.argv[1], "r") as config:
        config = json.load(config)
    train(config)