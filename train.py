import os
from datetime import datetime as dt
import tqdm
import cv2

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import RmbgDataset
from u2net import U2NET, U2NETP
from utils import fix_seed_torch, get_optimizer, get_loss, normPRED

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

    is_accumulation = cfg.getboolean('is_accumulation')
    if is_accumulation:
        accumulation_steps = cfg.getint('accumulation_steps')

    device = 'cuda' if torch.cuda.is_available() and cfg.getboolean('cuda') else 'cpu'

    t = dt.today().strftime("%Y%m%d%H%M")
    tensorboard_path = os.path.join('run', t)

    # Define Tensorboard
    writer = SummaryWriter(tensorboard_path)

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

    # Add network summary graph
    writer.add_graph(net, torch.rand(batch_size, 3, crop_shape[0], crop_shape[1]))

    if cfg.getboolean('multi_gpu') and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)
    net.to(device=device)

    optimizer = get_optimizer(net, 'adam', lr=lr)
    criterion = get_loss(cfg['loss'])

    iter_num = 0

    for epoc in range(epochs):
        net.train()
        net.zero_grad()
        for inputs, labels in tqdm.tqdm(train_dl, total=len(train_dl), mininterval=0.01):
            inputs = inputs.to(device)
            inputs.requires_grad = False
            labels = labels.to(device)
            labels.requires_grad = False

            optimizer.zero_grad()

            d0, d1, d2, d3, d4, d5, d6 = net(inputs)
            loss0, loss = criterion(d0, d1, d2, d3, d4, d5, d6, labels)


            if is_accumulation:
                loss = loss / accumulation_steps
                loss.backward()
                if (iter_num + 1) % accumulation_steps == 0:
                    optimizer.step()
                    net.zero_grad()
            else:
                loss.backward()
                optimizer.step()

            writer.add_scalar('train_loss0', loss0, iter_num)
            writer.add_scalar('train_loss', loss, iter_num)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], iter_num)

            if iter_num % 50 == 0:
                img = inputs[0]
                img = img / 2 + 0.5 #denormalize
                img = img.permute(1, 2, 0).numpy()
                pred = normPRED(d0[0])
                mask = pred.detach().permute(1, 2, 0).numpy()
                mask[mask >= 0.5] = 255
                mask[mask < 0.5] = 0
                not_mask = cv2.bitwise_not(mask)
                not_mask = cv2.cvtColor(not_mask, cv2.COLOR_GRAY2BGR)
                out = cv2.add(img, not_mask)
                writer.add_image('output_image', out, iter_num, dataformats='HWC')

            iter_num += 1

        net.eval()
        for inputs, labels in val_dl:
            inputs = inputs.to(device)
            inputs.requires_grad = False
            labels = labels.to(device)
            labels.requires_grad = False

            with torch.no_grad():
                d0, d1, d2, d3, d4, d5, d6 = net(inputs)
            loss0, loss = criterion(d0, d1, d2, d3, d4, d5, d6, labels)

            writer.add_scalar('eval_loss0', loss0, iter_num)
            writer.add_scalar('eval_loss', loss, iter_num)

    save_path = cfg['save_path']
    if save_path is not None:
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        torch.save(net.state_dict(), os.path.join(save_path, f'SSD_{t}.pth'))

    writer.close()