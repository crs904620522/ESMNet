import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class ResASPP3D(nn.Module):
    def __init__(self, channel, kernel_size, padding):
        super(ResASPP3D, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv3d(channel, channel, kernel_size=kernel_size, stride=1, padding=padding[0],
                                              dilation=1, bias=False), nn.ReLU(inplace=True))
        self.conv_2 = nn.Sequential(nn.Conv3d(channel, channel, kernel_size=kernel_size, stride=1, padding=padding[1],
                                              dilation=2, bias=False), nn.ReLU(inplace=True))
        self.conv_3 = nn.Sequential(nn.Conv3d(channel, channel, kernel_size=kernel_size, stride=1, padding=padding[2],
                                              dilation=4, bias=False), nn.ReLU(inplace=True))
        self.conv_t = nn.Conv3d(channel * 3, channel, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv_1(x))
        buffer_1.append(self.conv_2(x))
        buffer_1.append(self.conv_3(x))
        buffer_1 = self.conv_t(torch.cat(buffer_1, 1))
        return x + buffer_1

class ResidualBlock3D(nn.Module):
    expansion = 1
    def __init__(self, inchannel, outchannel, stride=1, kernel_size=3, padding=1, dilation=1):
        super(ResidualBlock3D, self).__init__()

        self.left = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding[0], bias=False),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv3d(outchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding[1], bias=False),
            nn.BatchNorm3d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm3d(outchannel)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResidualBlock2D(nn.Module):
    expansion = 1
    def __init__(self, inchannel, outchannel, stride=1, kernel_size=3, padding=1, dilation=1):
        super(ResidualBlock2D, self).__init__()

        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(outchannel)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class EPIFeatureExtraction(nn.Module):
    def __init__(self, in_channels, out_channels, epi_dim,kernel_size=3, padding=1,mode='0d', device=None):
        super(EPIFeatureExtraction, self).__init__()
        self.device = device
        self.mode = mode
        if self.mode=='0d':
            kernel_size = (1,3,3)
            padding = [(0,1,1),(0,1,1)]
            spp_kernel_size = (1,3,3)
            spp_padding = [(0,1,1),(0,2,2),(0,4,4)]
        elif self.mode=='90d':
            kernel_size=(3,1,3)
            padding = [(1,0,1),(1,0,1)]
            spp_kernel_size = (3,1,3)
            spp_padding = [(1,0,1),(2,0,2),(4,0,4)]
        else:
            kernel_size=(2,2,2)
            padding = [(1,1,1),(0,0,0)]
            spp_kernel_size = (3,3,3)
            spp_padding = [(1,1,1),(2,2,2),(4,4,4)]
        # 特征提取
        self.res1 = ResidualBlock3D(inchannel=in_channels, outchannel=16, kernel_size=kernel_size,
                                  padding=padding)
        self.res2 = ResidualBlock3D(inchannel=16, outchannel=32, kernel_size=kernel_size,
                                  padding=padding)
        self.res3 = ResidualBlock3D(inchannel=32, outchannel=48,kernel_size=kernel_size,
                                  padding=padding)
        self.spp = ResASPP3D(channel=96, kernel_size=spp_kernel_size,
                                  padding=spp_padding)
        self.res4 = ResidualBlock3D(inchannel=96, outchannel=out_channels,kernel_size=kernel_size,
                                  padding=padding)
        
        self.res5 = ResidualBlock3D(inchannel=96, outchannel=epi_dim,kernel_size=kernel_size,
                                  padding=padding)
    def forward(self, x):
        B, _, H, W,V = x.shape
        x0 = x
        x1 = self.res1(x0)
        x2 = self.res2(x1)
        x3 = self.res3(x2)
        tmp = torch.cat([x1,x2,x3],dim=1)
        sp = self.spp(tmp)
        x_cv = self.res4(sp)
        x_epi = self.res5(sp).reshape(B,-1,H,W,V)[:,:,:,:,int(V/2)]
        return x_cv, x_epi

class ViewAttention(nn.Module):
    def __init__(self, input_dim=16, output_dim=150, n_views=9, device=None):
        super(ViewAttention, self).__init__()
        self.n_views = n_views
        # local
        self.local_attention = nn.Sequential(
            nn.Conv3d(input_dim*self.n_views, 18, kernel_size=3,padding=1),
            nn.BatchNorm3d(18),
            nn.ReLU(inplace=True),
            nn.Conv3d(18, self.n_views, kernel_size=3,padding=1),
            nn.BatchNorm3d(self.n_views),
        )
        # global
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(input_dim*self.n_views, 18, kernel_size=3,padding=1),
            nn.BatchNorm3d(18),
            nn.ReLU(inplace=True),
            nn.Conv3d(18, self.n_views, kernel_size=3,padding=1),
            nn.BatchNorm3d(self.n_views),
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_last =  nn.Conv3d(input_dim * self.n_views, output_dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        B, M, C, H, W, N = x.shape
        x_l = self.local_attention(x.reshape(B,-1,H,W,N))
        x_g = self.global_attention(x.reshape(B,-1,H,W,N))
        xlg = x_l + x_g
        wei = self.sigmoid(xlg).reshape(B,M,1,H,W,N)  # B,1,9,H,W,N
        out = self.conv_last((wei*x).reshape(B,-1,H,W,N))
        return out

class EPICostConstrction(nn.Module):
    def __init__(self, input_dim=16,  output_dim=150, n_views=9, mode='0d', device=None):
        super(EPICostConstrction, self).__init__()

        self.device = device
        self.mode = mode
        self.n_views = n_views
        self.view_attention = ViewAttention(input_dim=input_dim, output_dim=output_dim, n_views=self.n_views, device=self.device)
        
    def forward(self, x):
        B, C, H, W, M = x.shape
        x = x.permute(4,0,1,2,3)
        view_list = list()
        for i in range(0, M):
            view_list.append(x[i, :])

        if self.mode == '0d':
            disparity_costs = list()
            for d in range(-4, 5):
                if d == 0:
                    tmp_list = list()
                    for i in range(len(view_list)):
                        tmp_list.append(view_list[i])
                else:
                    tmp_list = list()
                    for i in range(len(view_list)):
                        v, u = 4, i
                        rate = [ 2*d * (u - 4) / W, 2*d * (v - 4) / H]
                        theta = torch.tensor([[1, 0, rate[0]], [0, 1, rate[1]]], dtype=float).to(self.device)
                        grid = F.affine_grid(theta.unsqueeze(0).repeat(B, 1, 1), view_list[i].size()).type_as(view_list[i])
                        temp = F.grid_sample(view_list[i], grid)
                        tmp_list.append(temp)
                cost = torch.cat([i.reshape(B,1,C,H,W) for i in tmp_list], dim=1) # B, M, C, H, W
                disparity_costs.append(cost)
            cost_volume = torch.cat([i.reshape(B,M,C,H,W,1) for i in disparity_costs], dim=-1)
            cost_volume = cost_volume.reshape(B,M,C,H,W,9)
        elif self.mode == '90d':
            disparity_costs = list()
            for d in range(-4, 5):
                if d == 0:
                    tmp_list = list()
                    for i in range(len(view_list)):
                        tmp_list.append(view_list[i])
                else:
                    tmp_list = list()
                    for i in range(len(view_list)):
                        v, u = i, 4 
                        rate = [ 2*d * (u - 4) / W, 2*d * (v - 4) / H]
                        theta = torch.tensor([[1, 0, rate[0]], [0, 1, rate[1]]], dtype=float).to(self.device)
                        grid = F.affine_grid(theta.unsqueeze(0).repeat(B, 1, 1), view_list[i].size()).type_as(view_list[i])
                        temp = F.grid_sample(view_list[i], grid)
                        tmp_list.append(temp)
                cost = torch.cat([i.reshape(B,1,C,H,W) for i in tmp_list], dim=1) # B, M, C, H, W
                disparity_costs.append(cost)
            cost_volume = torch.cat([i.reshape(B,M,C,H,W,1) for i in disparity_costs], dim=-1)
            cost_volume = cost_volume.reshape(B,M,C,H,W,9)
        elif self.mode == '45d':
            disparity_costs = list()
            for d in range(-4, 5):
                if d == 0:
                    tmp_list = list()
                    for i in range(len(view_list)):
                        tmp_list.append(view_list[i])
                else:
                    tmp_list = list()
                    for i in range(len(view_list)):
                        v, u = i, self.n_views - i - 1, 
                        rate = [ 2*d * (u - 4) / W, 2*d * (v - 4) / H]
                        theta = torch.tensor([[1, 0, rate[0]], [0, 1, rate[1]]], dtype=float).to(self.device)
                        grid = F.affine_grid(theta.unsqueeze(0).repeat(B, 1, 1), view_list[i].size()).type_as(view_list[i])
                        temp = F.grid_sample(view_list[i], grid)
                        tmp_list.append(temp)
                cost = torch.cat([i.reshape(B,1,C,H,W) for i in tmp_list], dim=1) # B, M, C, H, W
                disparity_costs.append(cost)
            cost_volume = torch.cat([i.reshape(B,M,C,H,W,1) for i in disparity_costs], dim=-1)
            cost_volume = cost_volume.reshape(B,M,C,H,W,9)
        elif self.mode == '135d':
            disparity_costs = list()
            for d in range(-4, 5):
                if d == 0:
                    tmp_list = list()
                    for i in range(len(view_list)):
                        tmp_list.append(view_list[i])
                else:
                    tmp_list = list()
                    for i in range(len(view_list)):
                        v, u = i, i 
                        rate = [ 2*d * (u - 4) / W, 2*d * (v - 4) / H]
                        theta = torch.tensor([[1, 0, rate[0]], [0, 1, rate[1]]], dtype=float).to(self.device)
                        grid = F.affine_grid(theta.unsqueeze(0).repeat(B, 1, 1), view_list[i].size()).type_as(view_list[i])
                        temp = F.grid_sample(view_list[i], grid)
                        tmp_list.append(temp)
                cost = torch.cat([i.reshape(B,1,C,H,W) for i in tmp_list], dim=1) # B, M, C, H, W
                disparity_costs.append(cost)
            cost_volume = torch.cat([i.reshape(B,M,C,H,W,1) for i in disparity_costs], dim=-1)
            cost_volume = cost_volume.reshape(B,M,C,H,W,9)
        cv = self.view_attention(cost_volume).reshape(B,-1,H,W,9)
        return cv

class Aggregation(nn.Module):
    def __init__(self, views, input_dim=150, hidden_dim=150, device=None):
        super(Aggregation, self).__init__()
        self.device = device
        self.views = views

        self.conv1 = ResidualBlock3D(inchannel=input_dim,outchannel=hidden_dim,kernel_size=(3,3,3),padding=(1,1,1))
        self.conv2 = ResidualBlock3D(inchannel=hidden_dim,outchannel=hidden_dim,kernel_size=(3,3,3),padding=(1,1,1))
        self.conv3 = nn.Conv3d(hidden_dim, 1, kernel_size=3,padding=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x0 = x
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        feats = self.conv3(x2) # B,1,H,W,9
        weight = self.softmax(feats)
        B,_,H,W,N = weight.shape
        weight =weight.reshape(B,H,W,N)
        disparity_values = torch.linspace(-4, 4, 9).to(self.device)
        disparity_values = disparity_values.reshape(1, 1, 1, 9)
        disparity_values = disparity_values.repeat(B,H, W,1)
        depth_map = (disparity_values * weight).sum(dim=-1)
        
        feats = feats.permute(0,4,1,2,3).reshape(B,N,H,W)
        depth_map = depth_map.reshape(B,1,H,W)
        weight = weight.reshape(B,H,W,N)

        return feats, weight, depth_map

class Fusion(nn.Module):
    def __init__(self, input_dim=150, hidden_dim=150, device=None):
        super(Fusion, self).__init__()
        self.device = device

        self.conv1 = ResidualBlock2D(inchannel=input_dim,outchannel=hidden_dim,kernel_size=3,padding=1)
        self.conv2 = ResidualBlock2D(inchannel=hidden_dim,outchannel=hidden_dim,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, 9, kernel_size=3,padding=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x0 = x
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        feats = self.conv3(x2) # B,9,h,w
        weight = self.softmax(feats.permute(0,2,3,1))
        B,H,W,N = weight.shape
        weight =weight.reshape(B,H,W,N)
        disparity_values = torch.linspace(-4, 4, 9).to(self.device)
        disparity_values = disparity_values.reshape(1, 1, 1, 9)
        disparity_values = disparity_values.repeat(B,H, W,1)
        depth_map = (disparity_values * weight).sum(dim=-1)
        
        feats = feats.reshape(B,N,H,W)
        depth_map = depth_map.reshape(B,1,H,W)
        weight = weight.reshape(B,H,W,N)

        return feats, weight, depth_map

class Net(nn.Module):
    def __init__(self, input_dim,feats_dim=8, branch_dim=70, hidden_dim=100, n_views=9, device=None):
        super(Net, self).__init__()
        self.device = device
        self.n_views = n_views

        self.EPIFeature90d = EPIFeatureExtraction(in_channels=input_dim, out_channels=feats_dim, epi_dim=branch_dim, mode='90d')
        self.EPIFeature0d = EPIFeatureExtraction(in_channels=input_dim, out_channels=feats_dim,epi_dim=branch_dim, mode='0d')
        self.EPIFeature45d = EPIFeatureExtraction(in_channels=input_dim, out_channels=feats_dim,epi_dim=branch_dim, mode='45d')
        self.EPIFeature135d = EPIFeatureExtraction(in_channels=input_dim, out_channels=feats_dim,epi_dim=branch_dim, mode='135d')

        self.BuildCost0d = EPICostConstrction(input_dim=feats_dim,  output_dim=branch_dim,  n_views=self.n_views,mode='0d',device=self.device)
        self.BuildCost90d = EPICostConstrction(input_dim=feats_dim,  output_dim=branch_dim,  n_views=self.n_views,mode='90d',device=self.device)
        self.BuildCost45d = EPICostConstrction(input_dim=feats_dim,  output_dim=branch_dim,  n_views=self.n_views,mode='45d',device=self.device)
        self.BuildCost135d = EPICostConstrction(input_dim=feats_dim,  output_dim=branch_dim,  n_views=self.n_views,mode='135d',device=self.device)

        self.CVAttention = nn.Sequential(
            nn.Conv3d(4*branch_dim, branch_dim, kernel_size=1,padding=0),
            nn.BatchNorm3d(branch_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(branch_dim, 4, kernel_size=1,padding=0),
            nn.Sigmoid()
        )
        self.EPIAttention = nn.Sequential(
            nn.Conv2d(4*branch_dim,branch_dim, kernel_size=1,padding=0),
            nn.BatchNorm2d(branch_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_dim, 4, kernel_size=1,padding=0),
            nn.Sigmoid()
        )

        self.Aggregation = Aggregation(views=self.n_views, input_dim=4*branch_dim, hidden_dim=hidden_dim, device=self.device)
        self.Fusion = Fusion(input_dim=4*branch_dim, hidden_dim=hidden_dim, device=self.device)
        
        self.Regression = nn.Sequential(
            nn.Conv2d(18,9, kernel_size=3,padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(inplace=True),
            nn.Conv2d(9, 1, kernel_size=3,padding=1),
            nn.Sigmoid()
        )


    def forward(self, input,gt=None):
        B, N, H, W, V = input.shape
        input_0,  input_45, input_90, input_135 = input[:, 0], input[:, 1],  input[:, 2], input[:, 3]

        # 0
        B, H, W, V = input_0.shape  
        input_0 = input_0.reshape(B, 1, H, W, V)
        x_0, x_epi0 = self.EPIFeature0d(input_0)
        # 90
        B, H, W, V = input_90.shape
        input_90 = input_90.reshape(B, 1, H, W, V)
        x_90, x_epi90 = self.EPIFeature90d(input_90)
        # 45
        B, H, W, V = input_45.shape
        input_45 = input_45.reshape(B, 1, H, W, V)
        x_45, x_epi45 = self.EPIFeature45d(input_45)
        # 135
        B, H, W, V = input_135.shape
        input_135 = input_135.reshape(B, 1, H, W, V)
        x_135, x_epi135 = self.EPIFeature135d(input_135)
        # epi
        epi_feats = torch.cat([x_epi0,x_epi45,x_epi90,x_epi135],dim=1)

        # cost
        cv_0 = self.BuildCost0d(x_0)
        cv_90 = self.BuildCost90d(x_90)
        cv_45 = self.BuildCost45d(x_45)
        cv_135 = self.BuildCost135d(x_135)  
        cv_feats = torch.cat([cv_0,  cv_45, cv_90,  cv_135], dim=1)

        # cv aggregation
        B,_,H,W,N = cv_feats.shape
        cv_attention = self.CVAttention(cv_feats.reshape(B,-1,H,W,9)).reshape(B,4,H,W,N)
        cv = torch.cat([cv_0 * cv_attention[:,0:1],cv_45 * cv_attention[:,1:2], cv_90 * cv_attention[:,2:3], cv_135 * cv_attention[:,3:4]],dim=1) # B, 2*C, H, W, N
        cv_feats, cv_weight, cv_depth_map = self.Aggregation(cv)
        
        # epi aggregation
        epi_attention = self.EPIAttention(epi_feats.reshape(B,-1,H,W)).reshape(B,4,H,W)
        epi = torch.cat([x_epi0 * epi_attention[:,0:1],x_epi45 * epi_attention[:,1:2], x_epi90 * epi_attention[:,2:3], x_epi135 * epi_attention[:,3:4]],dim=1) # B, 2*C, H, W, N
        epi_feats, epi_weight, epi_depth_map = self.Fusion(epi)

        # regression    
        fusion_feats = torch.cat([cv_feats,epi_feats],dim=1)
        fusion_attention = self.Regression(fusion_feats)
        final_depth_map = fusion_attention * cv_depth_map + (1-fusion_attention) * epi_depth_map

        out = {'cv_weight':cv_weight,
               'epi_weight':epi_weight,
               'final_depth_map':final_depth_map.reshape(B,H,W,1),
               'gt':gt,
               'cv_depth_map': cv_depth_map.reshape(B,H,W,1),
               'epi_depth_map':epi_depth_map.reshape(B,H,W,1)
        }
        return out