import yaml
import os
import torch
import torch.nn as nn
from torchsummary import summary
from lf2disp.ESMNet.models.net import Net

class ESMNet(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        self.device = device
        input_dim = cfg['ESMNet']['encode']['input_dim']
        branch_dim = cfg['ESMNet']['encode']['branch_dim']
        hidden_dim = cfg['ESMNet']['encode']['hidden_dim']
        feats_dim = cfg['ESMNet']['encode']['feats_dim']
        n_views = cfg['data']['views']
        self.net = Net(input_dim=input_dim, branch_dim=branch_dim,hidden_dim=hidden_dim,feats_dim=feats_dim, n_views=n_views, device=self.device).to(self.device)

    def forward(self, input,gt=None):
        B1,B2,N,H,W,V = input.shape
        out = self.net(input.reshape(B1*B2,N,H,W,V),gt)
        return out

    def to(self, device):
        model = super().to(device)
        model._device = device
        return model