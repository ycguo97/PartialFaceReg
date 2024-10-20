import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import pickle
import cv2
from scipy.io import loadmat
import torch.nn.functional as F


from detect import detect
from torch.utils.data import Dataset

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


dir_checkpoint = 'checkpoints/'

class Mask(nn.Module):
    def __init__(self, weight, device):
        super(Mask, self).__init__()
        self.device = device
        self.weight = weight

    def forward(self, pred, true):
        term1 = true * torch.log(pred + 1e-6)
        term2 = (1 - true) * torch.log(1 - pred + 1e-6)
        error = -torch.sum(term1 + term2)
        return error

class Landmark(nn.Module):
    def __init__(self, weight, device):
        super(Landmark, self).__init__()
        self.device = device
        self.weight = weight

    def forward(self, points, true, mask):
        error = points - true
        error = error * torch.unsqueeze(mask[:, :, 1], 2)
        error = torch.mean(torch.sum(torch.abs(error), 2))
        return error

class Gen_data(Dataset):
    def __init__(self, imgs, lms, masks):
        self.imgs = imgs
        self.lms = lms
        self.masks = masks

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        imgs = self.imgs[idx, :, :]
        lms = self.lms[idx, :, :]
        masks = self.masks[idx, :, :]
        return imgs, lms, masks

def train_net(net,
              device,
              epochs=5,
              batch_size=10,
              weight = 0,
              lr=0.001,
              save_cp=True):

    data = np.load('./detect_data.npz')
    imgs = data['imgs']
    lms = data['lms']
    masks = data['masks']

    dataset = Gen_data(imgs, lms, masks)
    
    n_train = len(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
    writer = SummaryWriter(comment='LR_{}_BS_{}'.format(lr, batch_size))
    global_step = 0

    logging.info('''Starting training:
        Epochs:          {}
        Batch size:      {}
        Learning rate:   {}
        Checkpoints:     {}
        Device:          {}
        Weight:          {}
    '''.format(epochs, batch_size, lr, save_cp, device.type, weight))

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=0, momentum=0)

    lambda1 = lambda epoch: 0.95 ** (epoch/80.)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1)
    criterion1 = Mask(weight, device)
    criterion2 = Landmark(weight, device)

    current = 0
    for epoch in range(epochs):
        net.train()

        current += 1
        epoch_loss = 0
        sample_cnt = 0
        lm_loss = 0
        mask_loss = 0

        with tqdm(total=n_train, desc='Epoch {}/{}'.format(epoch+1, epochs), unit='img') as pbar:
            for batch in train_loader:
                imgs, lms, masks = batch

                imgs = imgs.to(device=device, dtype=torch.float32)
                imgs = torch.unsqueeze(imgs, 1)
                lms = lms.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.float32)

                pred_lms, pred_masks = net(imgs)

                loss1 = criterion1(pred_masks[:, :, 1], masks[:, :, 1])
                loss2 = criterion2(pred_lms, lms, masks)

                if current <= 50:
                    loss = loss2
                else:
                    loss = loss2 + 0.01 * (current-50) * loss1/(masks.shape[0]*masks.shape[1])

                epoch_loss += loss.item()
                lm_loss += loss2.item()
                mask_loss += loss1.item()

                sample_cnt += 1
                pbar.set_postfix(**{'lms': lm_loss/sample_cnt, 'masks': mask_loss/sample_cnt, 'epoch avg loss:': epoch_loss / sample_cnt})

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(batch_size)
                global_step += 1

                
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if (epoch+1) % 1 == 0:
                torch.save(net.state_dict(),
                           dir_checkpoint + 'CP_epoch{}.pth'.format(epoch + 1))
                logging.info('Checkpoint {} saved !'.format(epoch + 1))
    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the BSNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=4000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=10,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default = False,
                        help='Load model from a .pth file')
    parser.add_argument('-w', '--weight', dest='weight', type=float, default=1,
                        help='The weight of the custom loss')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    logging.info('Using device {}'.format(device))

    net = detect()

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info('Model loaded from {}'.format(args.load))

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  weight = args.weight,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
