import random

import torch
import numpy as np


def set_random_seed(seed):
    print("Random seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
