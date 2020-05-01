from module.metrics import Accuracy


def create_metrics(args):
    return {"Acc.": Accuracy()}
