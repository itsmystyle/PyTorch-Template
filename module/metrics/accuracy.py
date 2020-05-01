import numpy as np


class Accuracy(object):
    def __init__(self):
        self.name = "Acc."
        self.n = 0
        self.n_correct = 0

    def reset(self):
        self.n = 0
        self.n_correct = 0

    def update(self, preds, targets):
        preds = np.exp(preds)
        self.n_correct += (preds.argmax(axis=1) == targets).sum()
        self.n += preds.shape[0]

    def print_score(self):
        return "{:.5f}".format(self.get_score())

    def get_score(self):
        return self.n_correct / self.n
