import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from loss import multi_focal_loss_fusion, multi_bce_loss_fusion

def fix_seed_torch(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def get_optimizer(net, optim_name ='Adam', lr=1e-3, decay=5e-4, momentum=0.99):
    optim_name = optim_name.lower()
    if optim_name == 'adam':
        return optim.Adam(net.parameters(), lr=lr, weight_decay=decay)
    elif optim_name == 'sgd':
        return optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    else:
        raise ValueError(f'{optim_name} is not in this projects.')

def get_loss(loss_name='bce'):
    if loss_name == 'bce':
        return multi_bce_loss_fusion
    elif loss_name == 'focal':
        return multi_focal_loss_fusion
    else:
        raise ValueError(f'{loss_name} is not in this projects.')

# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn