import numpy as np

from module.trainer.trainer_base import TrainerBase
from module.loss.factory import create_criterions
from module.metrics.factory import create_metrics


class Trainer(TrainerBase):
    def __init__(self, **kwargs):
        super(Trainer, self).__init__(**kwargs)

    def set_criterion(self):
        self.criterion = create_criterions(self.num_class, self.device)

    def set_metrics(self):
        self.metrics = create_metrics()

    def _forward_one_iter(self, X, y, train=True):
        preds = self.model(X)
        loss = self.criterion(preds, y)

        self.metrics["Acc."].update(preds, y)

        postfix_dict = {"Acc.": self.metrics["Acc."].print_score()}

        if train:
            if "Loss" not in self.train_loss:
                self.train_loss["Loss"] = []
            self.train_loss["Loss"].append(loss.item())
            postfix_dict["Loss"] = f"{(np.mean(self.train_loss['Loss'])):.5f}"
        else:
            if "Loss" not in self.valid_loss:
                self.valid_loss["Loss"] = []
            self.valid_loss["Loss"].append(loss.item())
            postfix_dict["Loss"] = f"{(np.mean(self.valid_loss['Loss'])):.5f}"

        return loss, postfix_dict

    def check_best_model(self):
        if self.metrics["Acc."].score() > self.best_val:
            self.best_val = self.metrics["Acc."].score()
            return True
        else:
            return False

    def step_lr_scheduler(self):
        self.lr_scheduler.step(np.mean(self.valid_loss["Loss"]))
