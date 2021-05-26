import torch
import os
from torch.utils.data import DataLoader

from datasets import RmbgDataset
from u2net import U2NET, U2NETP
from utils import multi_bce_loss_fusion, multi_focal_loss_fusion, fix_seed_torch

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

    if cfg.getboolean('cuda') and torch.cuda.is_available():
        net.cuda()

    # TODO: train 코드 작성 및 accumulate 코드 작성