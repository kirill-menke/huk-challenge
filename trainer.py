import os
import logging
from typing import Tuple

import torch
from torch import Tensor
from numpy import ndarray
from sklearn.metrics import mean_squared_error
from ray import tune
from tqdm import tqdm


class Trainer:

    def __init__(self, model=None, loss=None, optim=None, train_dl=None, val_dl=None, scheduler=None, cuda=True, enable_log=True, checkpoint=False):
        self.model = model
        self.loss = loss
        self.optim = optim
        self.scheduler = scheduler
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.cuda = cuda
        self.checkpoint = checkpoint

        if cuda:
            self.model = model.cuda()
            self.loss = loss.cuda()

        if not enable_log:
            self.disable_tqdm = True
            logging.disable(logging.CRITICAL + 1)
        else:
            self.disable_tqdm = False
            logging.basicConfig(format="%(message)s", level="INFO")


    def save_checkpoint(self, epoch: int, rmse: float) -> None:
        """ Save a checkpoint of the model and optimizer state """
        checkpoints_path = os.path.join(os.environ["HUK_CHAL"], "checkpoints")
        os.makedirs(checkpoints_path, exist_ok=True)

        state = {"model_state_dict": self.model.state_dict(), "optim_state_dict": self.optim.state_dict(), "epoch": epoch, "rmse": rmse}
        torch.save(state, os.path.join(checkpoints_path, f"checkpoint_{epoch}_{rmse}.pth"))
            

    def train_step(self, x: Tensor, y: Tensor) -> Tensor:
        """ Perform a single training step (one batch) """
        self.model.zero_grad()

        y_preds = self.model(x)
        loss = torch.sqrt(self.loss(y_preds, y))
        loss.backward()
        self.optim.step()

        return loss
        
        
    def train(self) -> float:
        """ Perform a single training epoch """
        self.model.train()
        total_loss = 0

        for x, y in self.train_dl:
            x, y = (x.cuda(), y.cuda()) if self.cuda else (x, y)
            total_loss += self.train_step(x, y)
        
        if self.scheduler:
            self.scheduler.step()

        avg_loss = total_loss / len(self.train_dl)
        return avg_loss.item()
    

    @torch.no_grad()
    def val_step(self, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """ Perform a single validation step (one batch) """
        y_preds = self.model(x)
        loss = torch.sqrt(self.loss(y_preds, y))
        return loss, y_preds
    

    def validate(self) -> Tuple[ndarray, ndarray, float]:
        """ Perform validation on the entire validation set """
        self.model.eval()
        
        y_preds = []
        y_grounds = []

        total_loss = 0

        for x, y in self.val_dl:
            x, y = (x.cuda(), y.cuda()) if self.cuda else (x, y)
            loss, y_pred = self.val_step(x, y)
            total_loss += loss
        
            y_preds.append(y_pred.cpu())
            y_grounds.append(y.cpu())
        
        y_preds = torch.cat(y_preds).numpy()
        y_grounds = torch.cat(y_grounds).numpy()

        avg_loss = total_loss / len(self.val_dl)
        return y_preds, y_grounds, avg_loss.item()


    def calculate_metrics(self, y_preds: Tensor, y_grounds: Tensor) -> Tuple[float, float, float, float]:
        """ Calculate the RMSE for the entire validation set """
        # Calculate RMSE for all samples
        rmse_hood, rmse_backdoor_left = mean_squared_error(y_grounds, y_preds, multioutput='raw_values', squared=False)

        # Calculate RMSE for samples with non-zero targets only
        target_hood, target_backdoor_left = y_grounds[:, 0][y_grounds[:, 0] != 0], y_grounds[:, 1][y_grounds[:, 1] != 0]
        pred_hood, pred_backdoor_left = y_preds[:, 0][y_grounds[:, 0] != 0], y_preds[:, 1][y_grounds[:, 1] != 0]
        non_zero_rmse_hood = mean_squared_error(target_hood, pred_hood, squared=False)
        non_zero_rmse_backdoor_left = mean_squared_error(target_backdoor_left, pred_backdoor_left, squared=False)
        
        return rmse_hood, rmse_backdoor_left, non_zero_rmse_hood, non_zero_rmse_backdoor_left


    def fit(self, epochs: int) -> None:
        """ Train the model for the given number of epochs """
        
        for epoch in tqdm(range(1, epochs + 1), disable=self.disable_tqdm):
            logging.info(f"Epoch {epoch}: ")

            train_loss = self.train()
            logging.info(f"Train loss: {train_loss:.4f}")

            y_preds, y_grounds, val_loss = self.validate()
            rmse_hood, rmse_backdoor_left, non_zero_rmse_hood, non_zero_rmse_backdoor_left = self.calculate_metrics(y_preds, y_grounds)

            avg_rmse = (rmse_hood + rmse_backdoor_left) / 2
            non_zero_rmse = (non_zero_rmse_hood + non_zero_rmse_backdoor_left) / 2

            logging.info(f"Val loss: {val_loss:.4f}")
            logging.info(f"Mean RMSE: {avg_rmse:.4f}, RMSE hood: {rmse_hood:.4f}, RMSE backdoor left: {rmse_backdoor_left:.4f}")

            # Report metrics to raytune during HPO
            if tune.is_session_enabled():
                metrics = {"avg_rmse": avg_rmse, "rmse_hood": rmse_hood, "rmse_backdoor_left": rmse_backdoor_left,
                           "non_zero_rmse": non_zero_rmse, "non_zero_rmse_hood": non_zero_rmse_hood, "non_zero_rmse_backdoor_left": non_zero_rmse_backdoor_left,
                           "train_loss": train_loss, "val_loss": val_loss}
                tune.report(**metrics)

            if self.checkpoint and avg_rmse < 0.11:
                self.save_checkpoint(epoch, avg_rmse)