import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
import time
import os
import cv2
from lf2disp.utils.utils import depth_metric, write_pfm, LFdivide, LFintegrate
from einops import rearrange



class GeneratorDepth(object):

    def __init__(self, model, cfg=None, device=None):
        self.model = model.to(device)
        self.device = device
        self.generate_dir = cfg['generation']['generation_dir']
        self.name = cfg['generation']['name']

        if not os.path.exists(self.generate_dir):
            os.makedirs(self.generate_dir)

    def generate_depth(self, data, id=0):
        ''' Generates the output depthmap
        '''
        self.model.eval()
        device = self.device
        image = data.get('image').to(device)
        label = data.get('label').to(device)
        B1, B2, N, H, W, M  = image.shape
        label = label.reshape(B1 * B2, H, W,-1)
        if H>512 or W>512:
            # crop预测
            patchsize = 128 # 128
            stride = patchsize // 2
            data = image.reshape(N,H,W,M,1).permute(1,2,0,3,4)# H W N M,1
            sub_lfs = LFdivide(data.permute(3,4,2,0,1), patchsize, stride)
            n1, n2, u, v, nc, h, w = sub_lfs.shape
            sub_lfs = rearrange(sub_lfs, 'n1 n2 u v nc h w -> (n1 n2) h w nc u v')     
            mini_batch = 12
            num_inference = (n1 * n2) // mini_batch  
            # 数据准备
            out_disp = []
            for idx_inference in range(num_inference):
                current_lfs = sub_lfs[idx_inference * mini_batch: (idx_inference + 1) * mini_batch, :, :, :, :, :] # b h w nc u v
                current_lfs = rearrange(current_lfs, 'b h w nc u v -> b nc h w (u v)')
                with torch.no_grad():
                    temp = self.model(current_lfs.unsqueeze(0))['final_depth_map']
                temp = temp.reshape(-1, 1, patchsize, patchsize)
                out_disp.append(temp)
            if (n1 * n2) % mini_batch:
                current_lfs = sub_lfs[(idx_inference + 1) * mini_batch:, :, :, :, :, :]
                current_lfs = rearrange(current_lfs, 'b h w nc u v -> b nc h w (u v)')
                with torch.no_grad():
                    temp = self.model(current_lfs.unsqueeze(0))['final_depth_map']
                temp = temp.reshape(-1, 1, patchsize, patchsize)
                out_disp.append(temp)
            out_disps = torch.cat(out_disp, dim=0)
            out_disps = rearrange(out_disps, '(n1 n2) c h w -> n1 n2 c h w', n1=n1, n2=n2)
            disp = LFintegrate(out_disps, patchsize, patchsize // 2)
            # disp取值
            depthmap = disp[0: H, 0: W]
            cv_depth_map = depthmap
            epi_depth_map = depthmap
        else:
            with torch.no_grad():
                out= self.model(image)
                depthmap = out['final_depth_map']
                cv_depth_map = out['cv_depth_map']
                epi_depth_map = out['epi_depth_map']
            
        depthmap = depthmap.cpu().numpy().reshape(B1 * B2, H, W, 1)[0]
        label = label.cpu().numpy().reshape(B1 * B2, H, W, 1)[0]
        cv_depth_map = cv_depth_map.cpu().numpy().reshape(B1 * B2, H, W, 1)[0]
        epi_depth_map = epi_depth_map.cpu().numpy().reshape(B1 * B2, H, W, 1)[0]

        metric = depth_metric(label[15:-15, 15:-15], depthmap[15:-15, 15:-15])
        metric['id'] = id
        print("-------------------------------------------------------------\n")
        print('                            ' + str(id) + '                         \n ')
        print('cv_result',depth_metric(label[15:-15, 15:-15], depthmap[15:-15, 15:-15]))
        print('epi_result',depth_metric(label[15:-15, 15:-15], depthmap[15:-15, 15:-15]))
        print('final_result:', metric)

        depthpath = os.path.join(self.generate_dir, self.name[id] + '.png')
        labelpath = os.path.join(self.generate_dir, self.name[id] + '_label.png')
        depth_fix = np.zeros((512, 512, 1), dtype=float)
        depth_fix = depthmap
        pfm_path = os.path.join(self.generate_dir, self.name[id] + '.pfm')
        write_pfm(depth_fix, pfm_path, scale=1.0)

        coarse = (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min())
        label = (label - label.min()) / (label.max() - label.min())
        cv2.imwrite(depthpath, coarse.copy() * 255.0)
        print('save coarse depth map in', depthpath)
        cv2.imwrite(labelpath, label.copy() * 255.0)
        print('save label in', labelpath)

        return metric


