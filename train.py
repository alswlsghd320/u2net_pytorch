import os
from datetime import datetime as dt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import RmbgDataset
from u2net import U2NET, U2NETP
from utils import fix_seed_torch, get_optimizer, get_loss

# Fix seed
fix_seed_torch(42)

def train(cfg):
    # Set config
    train_img_path = cfg['X_TRAIN_PATH']
    train_mask_path = cfg['Y_TRAIN_PATH']
    val_img_path = cfg['X_VAL_PATH']
    val_mask_path = cfg['Y_VAL_PATH']

    resize_shape = eval(cfg.get("resize_shape"), {}, {})
    crop_shape = eval(cfg.get("crop_shape"), {}, {})

    batch_size = cfg.getint('batch_size')
    epochs = cfg.getint('epochs')
    lr = cfg.getfloat('LR')

    device = 'cuda' if torch.cuda.is_available() and cfg.getboolean('cuda') else 'cpu'

    # Define Dataset, Dataloader
    train_ds = RmbgDataset(cfg, train_img_path, train_mask_path,
                           resize_shape, crop_shape, is_train=True)
    val_ds = RmbgDataset(cfg, val_img_path, val_mask_path,
                           resize_shape, crop_shape, is_train=True)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=8)

    # Define model
    model = cfg['model']
    if model == 'u2net':
        net = U2NET(3, 1)
    elif model == 'u2netp':
        net = U2NETP(3, 1)
    else:
        ValueError("model name must be 'u2net' or 'u2netp'")

    if cfg.getboolean('multi_gpu') and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device=device)

    optimizer = get_optimizer(net, 'adam')
    criterion = get_loss('focal')

    loss_list = []
    loss0_list = []
    iter_num = 0

    for epoc in range(epochs):
        net.train()
        for inputs, labels in train_dl:
            inputs = inputs.to(device)
            inputs.requires_grad = False
            labels = labels.to(device)
            labels.requires_grad = False

            optimizer.zero_grad()

            d0, d1, d2, d3, d4, d5, d6 = net(inputs)

            loss0, loss = criterion(d0, d1, d2, d3, d4, d5, d6, labels)

            loss.backward()
            optimizer.step()

            loss0_list.append(loss0.item())
            loss_list.append(loss.item())

        # for inputs, labels in val_dl:
        #     net.eval()
        #     inputs = inputs.to(device)
        #     inputs.requires_grad = False
        #     labels = labels.to(device)
        #     labels.requires_grad = False
        #
        #     with torch.no_grad():
        #         d0, d1, d2, d3, d4, d5, d6 = net(inputs)

    save_path = cfg['save_path']
    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        torch.save(net.state_dict(), os.path.join(save_path, f'SSD_{dt.today().strftime("%Y%m%d%H%M")}.pth'))


    # TODO: accumulate 코드 작성, 텐서보드