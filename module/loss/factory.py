import torch.nn as nn

from module.loss import CrossEntropyLabelSmooth


def create_criterions(args, num_classes, device=None):
    if args.smoothing:
        criterion = CrossEntropyLabelSmooth(num_classes, device=device)
    else:
        criterion = nn.NLLLoss()

    return criterion
