import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from lf2disp.training import BaseTrainer
import torch.nn as nn
import cv2
import math
from lf2disp.utils.utils import depth_metric
import numpy as np
import random
from scipy.stats import truncnorm


class Trainer(BaseTrainer):

    def __init__(self, model, optimizer, criterion=nn.MSELoss, device=None, cfg=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion()
        self.vis_dir = cfg['vis']['vis_dir']
        self.test_dir = cfg['test']['test_dir']

        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        print("use model:", self.model)
        print("use loss:", self.criterion)

    def train_step(self, data, iter=0):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data, imgid=0):
        device = self.device
        self.model.eval()
        image = data.get('image').to(device)
        label = data.get('label').to(device)
        B1, B2, N, H, W, M  = image.shape

        label = label.reshape(B1 * B2, H, W,-1)
        with torch.no_grad():
            out= self.model(image)
            depthmap = out['final_depth_map']
            
        depthmap = depthmap.cpu().numpy().reshape(B1 * B2, H, W, 1)[0][15:-15, 15:-15]
        label = label.cpu().numpy().reshape(B1 * B2, H, W, 1)[0][15:-15, 15:-15]

        metric = depth_metric(label, depthmap)
        metric['id'] = imgid
        return metric

    def visualize(self, data, id=0, vis_dir=None):
        self.model.eval()
        device = self.device
        if vis_dir is None:
            vis_dir = self.vis_dir
        image = data.get('image').to(device)
        label = data.get('label').to(device)
        B1, B2, N, H, W, M  = image.shape

        label = label.reshape(B1 * B2, H, W)
        with torch.no_grad():
            out= self.model(image)
            depthmap = out['final_depth_map']
        depthmap = depthmap.cpu().numpy().reshape(B1 * B2, H, W, 1)[0][15:-15, 15:-15]
        label = label.cpu().numpy().reshape(B1 * B2, H, W, 1)[0][15:-15, 15:-15]

        depthmap = (depthmap - label.min()) / (label.max() - label.min())
        label = (label - label.min()) / (label.max() - label.min())

        path = os.path.join(vis_dir, str(id) + '_.png')
        labelpath = os.path.join(vis_dir, '%03d_label.png' % id)

        cv2.imwrite(path, depthmap.copy() * 255.0)
        print('save depth map in', path)
        cv2.imwrite(labelpath, label.copy() * 255.0)
        print('save label in', labelpath)

    def compute_loss(self, data):
        device = self.device
        image = data.get('image').to(device)
        B1, B2, N, H, W, M = image.shape
        label = data.get('label').reshape(B1 * B2, H, W).to(device)
        out= self.model(image,label)

        # loss mae
        label = out['gt'].reshape(B1 * B2, H, W)
        depth_map = out['final_depth_map'].reshape(B1*B2,-1)
        loss_mae = self.criterion(depth_map.reshape(B1*B2,-1),label.reshape(B1*B2,-1)).mean()

        # loss align
        cv_res = out['cv_weight'].reshape(B1*B2,H,W,-1) # B,H,W,9
        epi_res = out['epi_weight'].reshape(B1*B2,H,W,-1) # B,H,W,9
        res_label = torch.ones_like(cv_res).to(device)
        res_label = res_label * label.reshape(B1*B2,H,W,1) # B, H, W, 9 
        x = torch.from_numpy(np.array([-4,-3,-2,-1,0,1,2,3,4])).reshape(1,1,1,9).to(device)
        miu = res_label
        delta = 0.5
        gauss =  torch.exp(-1*((x-miu)**2) / (2*(delta**2)))
        prob_label = gauss /  gauss.sum(-1).reshape(B1*B2,H,W,1)
        loss_align = self.criterion(cv_res.reshape(B1*B2,-1),prob_label.reshape(B1*B2,-1)).mean() + self.criterion(epi_res.reshape(B1*B2,-1),prob_label.reshape(B1*B2,-1)).mean()

        total_loss = loss_mae + 0.5*loss_align

        return total_loss
