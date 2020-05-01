import os
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter

from module.utils import set_random_seed


class TrainerBase(ABC):
    def __init__(
        self,
        args,
        train_dataloader,
        valid_dataloader,
        model,
        device=torch.device("cpu"),
        patience=5,
    ):
        self.args = args
        self.device = device

        # data
        self.num_class = train_dataloader.dataset.num_classes
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        # model
        self.model = model
        self.model.to(self.device)

        # optimizer
        self.set_optimizer()

        # metric
        self.set_metrics()
        self.best_val = 0.0
        self.best_train = 0.0

        # early stopping
        self.patience = patience
        self.es_count = self.patience

        # save dir
        self.save_path = os.path.join(self.args.model_dir, "model_best.pkl")
        if not os.path.isdir(self.args.model_dir):
            os.mkdir(self.args.model_dir)

        # tensorboardX writer
        self.writer = SummaryWriter(os.path.join(self.args.model_dir, "train_logs"))

        # set random seed
        set_random_seed(self.args.random_seed)

    @abstractmethod
    def set_criterion(self):
        pass

    @abstractmethod
    def set_metrics(self):
        pass

    @abstractmethod
    def _forward_one_iter(self, X, y, train=True):
        pass

    @abstractmethod
    def check_best_model(self):
        pass

    @abstractmethod
    def step_lr_scheduler(self):
        pass

    def set_optimizer(self):
        self.optim = optim.Adam(self.model.params(), lr=self.args.lr)

    def set_lr_scheduler(self, epoch):
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, factor=0.3, patience=3, verbose=True
        )

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    def fit(self, epoch=50):
        self.set_lr_scheduler(epoch)

        for e in range(epoch):
            print(f"Epoch :{e+1}")
            self._train_one_epoch()
            self.writer.add_scalars(
                "Loss", {"train": np.mean(self.train_loss["Loss"])}, e,
            )
            self.writer.add_scalars(
                "Accuracy", {"train": self.metrics["Acc."].get_score()}, e,
            )
            self._valid_one_epoch()
            self.writer.add_scalars(
                "Loss", {"valid": np.mean(self.valid_loss["Loss"])}, e,
            )
            self.writer.add_scalars(
                "Accuracy", {"valid": self.metrics["Acc."].get_score()}, e,
            )
            self.step_lr_scheduler()

            if self.es_count == 0:
                print(
                    f"Early stopping! Model performance has not increase for {self.patience} epoch."
                )
                break

    def _train_one_epoch(self):
        self.model.train()

        self.reset_metrics()
        self.train_loss = {}

        trange = tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader))
        for idx, batch in trange:
            X, y = batch["X"].to(self.device), batch["y"].to(self.device)

            loss, postfix_dict = self._forward_one_iter(X, y)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            trange.set_postfix(**postfix_dict)

    def _valid_one_epoch(self):
        self.model.eval()

        self.reset_metrics()
        self.valid_loss = {}

        trange = tqdm(enumerate(self.valid_dataloader), total=len(self.valid_dataloader))
        with torch.no_grad():
            for idx, batch in trange:
                X, y = batch["X"].to(self.device), batch["y"].to(self.device)

                _, postfix_dict = self._forward_one_iter(X, y, train=False)

                trange.set_postfix(**postfix_dict)

        self.es_count -= 1

        if self.check_best_model():
            self.es_count = self.patience
            self.save_model(self.save_path)
            print(f"Model saved!")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
