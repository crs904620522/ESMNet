# coding:utf-8
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import os
import cv2
import csv
from PIL import Image
import random
from lf2disp.utils import utils
import imageio
from skimage import io
import time

np.random.seed(160)


class HCInew(Dataset):
    def __init__(self, cfg, mode='train'):
        super(HCInew, self).__init__()
        self.datadir = cfg['data']['path']
        self.mode = mode
        if mode == 'train':
            self.imglist = []
            self.batch_size = cfg['training']['image_batch_size']
            with open(os.path.join(self.datadir, 'onlytrain.txt'), "r") as f:
                for line in f.readlines():
                    imgdir = line.strip("\n")
                    self.imglist.append(os.path.join(self.datadir, imgdir))
            self.number = len(self.imglist)
            self.views = cfg['data']['views']
            self.inputsize = cfg['training']['input_size']
            self.augmentation = cfg['training']['augmentation']
            self.transform = cfg['training']['transform']
        elif mode == 'test':  # val or test
            self.imglist = []
            self.batch_size = cfg['test']['image_batch_size']
            datafile = os.path.join(self.datadir, 'test.txt')
            with open(datafile, "r") as f:
                for line in f.readlines():
                    imgdir = line.strip("\n")
                    self.imglist.append(os.path.join(self.datadir, imgdir))
            self.number = len(self.imglist)
            self.views = cfg['data']['views']
            self.inputsize = cfg['test']['input_size']
            self.transform = cfg['test']['transform']
        elif mode == 'vis':  # vis
            self.imglist = []
            self.batch_size = cfg['vis']['image_batch_size']
            datafile = os.path.join(self.datadir, 'vis.txt')
            with open(datafile, "r") as f:
                for line in f.readlines():
                    imgdir = line.strip("\n")
                    self.imglist.append(os.path.join(self.datadir, imgdir))
            self.number = len(self.imglist)
            self.views = cfg['data']['views']
            self.inputsize = cfg['vis']['input_size']
            self.transform = cfg['vis']['transform']
        elif mode == 'generate':  # vis
            self.imglist = []
            self.batch_size = 1
            datafile = os.path.join(self.datadir, 'generate.txt')
            with open(datafile, "r") as f:
                for line in f.readlines():
                    imgdir = line.strip("\n")
                    self.imglist.append(os.path.join(self.datadir, imgdir))
            self.number = len(self.imglist)
            self.views = cfg['data']['views']
            self.inputsize = cfg['generation']['input_size']
            self.transform = cfg['generation']['transform']

        self.invalidpath = []
        with open(os.path.join(self.datadir, 'invalid.txt'), "r") as f:
            for line in f.readlines():
                imgpath = line.strip("\n")
                self.invalidpath.append(os.path.join(self.datadir, imgpath))

        self.traindata_all = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
        self.traindata_label = np.zeros((len(self.imglist), 512, 512), np.float32)
        self.boolmask_data = np.zeros((len(self.invalidpath), 512, 512), np.float32)
        # 图片预先加载好
        self.imgPreloading()

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        if self.mode == 'train':
            start = time.time()
            image, label = self.train_data()
            mask = np.ones_like(label)
            if self.augmentation:
                image, label, mask = self.data_aug(image, label)
        else: 
            image, label = self.val_data(idx)
            mask = np.ones_like(label)
        out = {'image': np.float32(np.clip(image,0.0,1.0)),
               'label': np.float32(label),
               'mask':mask,
               }
        return out

    def imgPreloading(self):
        for idx in range(0, len(self.imglist)):
            imgdir = self.imglist[idx]
            for i in range(0, self.views ** 2):
                imgname = 'input_Cam' + str(i).zfill(3) + '.png'
                imgpath = os.path.join(imgdir, imgname)
                img = np.uint8(imageio.imread(imgpath))
                self.traindata_all[idx, :, :, i // 9, i - 9 * (i // 9), :] = img
            labelname = 'gt_disp_lowres.pfm'
            labelpath = os.path.join(imgdir, labelname)
            if os.path.exists(labelpath):
                imgLabel = utils.read_pfm(labelpath)
            else:
                imgLabel = np.zeros((512, 512))
            self.traindata_label[idx] = imgLabel
        for idx in range(0, len(self.invalidpath)):
            boolmask_img = np.uint8(imageio.imread(self.invalidpath[idx]))
            boolmask_img = 1.0 * boolmask_img[:, :, 3] > 0
            self.boolmask_data[idx] = boolmask_img

        if self.mode == 'train':
            self.traindata_all_add1 = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
            self.traindata_all_sub1 = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
            self.traindata_all_add2 = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
            self.traindata_all_sub2 = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
            center = int(self.views / 2)
            for batch_i in range(0, len(self.imglist)):
                for v in range(0, self.views):  # v
                    for u in range(0, self.views):  # u
                        offsety = (center - v)
                        offsetx = (center - u)
                        mat_translation = np.float32([[1, 0, 1 * offsetx], [0, 1, 1 * offsety]])
                        self.traindata_all_add1[batch_i, :, :, v, u, :] = cv2.warpAffine(
                            self.traindata_all[batch_i, :, :, v, u, :],
                            mat_translation,
                            (512, 512))
                        mat_translation = np.float32([[1, 0, -1 * offsetx], [0, 1, -1 * offsety]])
                        self.traindata_all_sub1[batch_i, :, :, v, u, :] = cv2.warpAffine(
                            self.traindata_all[batch_i, :, :, v, u, :],
                            mat_translation,
                            (512, 512))
                        
                        mat_translation = np.float32([[1, 0, 2 * offsetx], [0, 1, 2 * offsety]])
                        self.traindata_all_add2[batch_i, :, :, v, u, :] = cv2.warpAffine(
                            self.traindata_all[batch_i, :, :, v, u, :],
                            mat_translation,
                            (512, 512))
                        mat_translation = np.float32([[1, 0, -2 * offsetx], [0, 1, -2 * offsety]])
                        self.traindata_all_sub2[batch_i, :, :, v, u, :] = cv2.warpAffine(
                            self.traindata_all[batch_i, :, :, v, u, :],
                            mat_translation,
                            (512, 512))

        print('imgPreloading', self.boolmask_data.shape, self.traindata_all.shape, self.traindata_label.shape)
        return

    def train_data(self):

        """ initialize image_stack & label """
        batch_size = self.batch_size
        label_size, input_size = self.inputsize, self.inputsize

        traindata_batch_90d = np.zeros((batch_size, input_size, input_size, self.views),
                                       dtype=np.float32)
        traindata_batch_0d = np.zeros((batch_size, input_size, input_size, self.views),
                                      dtype=np.float32)
        traindata_batch_45d = np.zeros((batch_size, input_size, input_size, self.views),
                                       dtype=np.float32)
        traindata_batch_m45d = np.zeros((batch_size, input_size, input_size, self.views),
                                        dtype=np.float32)

        traindata_batch_label = np.zeros((batch_size, label_size, label_size))

        """ inital variable """
        start1 = 0
        end1 = self.views - 1
        crop_half = int(0.5 * (input_size - label_size))

        """ Generate image stacks """
        for ii in range(0, batch_size):
            sum_diff = 0
            valid = 0

            while sum_diff < 0.01 * input_size * input_size or valid < 1:

                """//Variable for gray conversion//"""
                rand_3color = 0.05 + np.random.rand(3)
                rand_3color = rand_3color / np.sum(rand_3color)
                R = rand_3color[0]
                G = rand_3color[1]
                B = rand_3color[2]
                """
                    We use totally 16 LF images,(0 to 15) 
                    Since some images(4,6,15) have a reflection region, 
                    We decrease frequency of occurrence for them. 
                    Details in our epinet paper.
                # """

                aa_arr = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                                   0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                                   0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ])

                image_id = np.random.choice(aa_arr)
                traindata_all, traindata_label = self.choose_delta(image_id)
                """
                    //Shift augmentation for 7x7, 5x5 viewpoints,.. //
                    Details in our epinet paper.
                """
                if self.views == 7:
                    print("警告 还没有完善非9张视角的选项")
                    ix_rd = np.random.randint(0, 3) - 1
                    iy_rd = np.random.randint(0, 3) - 1
                if self.views == 9:
                    ix_rd = 0
                    iy_rd = 0

                kk = np.random.randint(17)
                if (kk < 8):
                    scale = 1
                elif (kk < 14):
                    scale = 2
                elif (kk < 17):
                    scale = 3
                idx_start = np.random.randint(0, 512 - scale * input_size)  # random_size
                idy_start = np.random.randint(0, 512 - scale * input_size)
                valid = 1

                """
                    boolmask: reflection masks for images(4,6,15)
                """

                # 这是去除高光
                if image_id in [4, 6, 15]:
                    if image_id == 4:
                        a_tmp = self.boolmask_data[0]
                    if image_id == 6:
                        a_tmp = self.boolmask_data[1]
                    if image_id == 15:
                        a_tmp = self.boolmask_data[2]

                    if (np.sum(a_tmp[
                               idx_start + scale * crop_half: idx_start + scale * crop_half + scale * label_size:scale,
                               idy_start + scale * crop_half: idy_start + scale * crop_half + scale * label_size:scale]) > 0
                            or np.sum(a_tmp[
                                      idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                                      idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale]) > 0):
                        valid = 0
                if valid > 0:
                    seq0to8 = np.array([i for i in range(self.views)]) + ix_rd
                    seq8to0 = np.array([i for i in range(self.views - 1, -1, -1)]) + iy_rd

                    image_center = (1 / 255) * np.squeeze(
                        R * traindata_all[
                            idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                            idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                            4 + ix_rd, 4 + iy_rd, 0].astype(
                            'float32') +
                        G * traindata_all[
                            idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                            idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                            4 + ix_rd, 4 + iy_rd, 1].astype(
                            'float32') +
                        B * traindata_all[
                            idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                            idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                            4 + ix_rd, 4 + iy_rd, 2].astype('float32'))
                    sum_diff = np.sum(
                        np.abs(image_center - np.squeeze(image_center[int(0.5 * input_size), int(0.5 * input_size)])))


                    # 这是转为灰度图了
                    traindata_batch_0d[ii, :, :, :] = np.squeeze(
                        R * traindata_all[
                            idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                            idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                            4 + ix_rd, seq0to8.tolist(), 0].astype(
                            'float32') +
                        G * traindata_all[
                            idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                            idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                            4 + ix_rd, seq0to8.tolist(), 1].astype(
                            'float32') +
                        B * traindata_all[
                            idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                            idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                            4 + ix_rd, seq0to8.tolist(), 2].astype(
                            'float32'))

                    traindata_batch_90d[ii, :, :, :] = np.squeeze(
                        R * traindata_all[
                            idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                            idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                            seq8to0.tolist(), 4 + iy_rd, 0].astype(
                            'float32') +
                        G * traindata_all[
                            idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                            idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                            seq8to0.tolist(), 4 + iy_rd, 1].astype(
                            'float32') +
                        B * traindata_all[
                            idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                            idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                            seq8to0.tolist(), 4 + iy_rd, 2].astype(
                            'float32'))
                    for kkk in range(start1, end1 + 1):
                        traindata_batch_45d[ii, :, :, kkk - start1] = np.squeeze(
                            R * traindata_all[
                                idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                                idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                                (8) - kkk + ix_rd, kkk + iy_rd,
                                0].astype(
                                'float32') +
                            G * traindata_all[
                                idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                                idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                                (8) - kkk + ix_rd, kkk + iy_rd,
                                1].astype(
                                'float32') +
                            B * traindata_all[
                                idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                                idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                                (8) - kkk + ix_rd, kkk + iy_rd,
                                2].astype(
                                'float32'))

                        traindata_batch_m45d[ii, :, :, kkk - start1] = np.squeeze(
                            R * traindata_all[
                                idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                                idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                                kkk + ix_rd, kkk + iy_rd, 0].astype(
                                'float32') +
                            G * traindata_all[
                                idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                                idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                                kkk + ix_rd, kkk + iy_rd, 1].astype(
                                'float32') +
                            B * traindata_all[
                                idx_start + scale * crop_half: idx_start + scale * crop_half + scale * input_size:scale,
                                idy_start + scale * crop_half: idy_start + scale * crop_half + scale * input_size:scale,
                                kkk + ix_rd, kkk + iy_rd, 2].astype(
                                'float32'))
                    '''
                     traindata_batch_label  <-- scale_factor*traindata_label[random_index, scaled_label_size, scaled_label_size] 
                     '''
                    if len(traindata_label.shape) == 5:
                        traindata_batch_label[ii, :, :] = (1.0 / scale) * traindata_label[
                                                                          idx_start + scale * crop_half: idx_start + scale * crop_half + scale * label_size:scale,
                                                                          idy_start + scale * crop_half: idy_start + scale * crop_half + scale * label_size:scale,
                                                                          4 + ix_rd, 4 + iy_rd]
                    else:
                        traindata_batch_label[ii, :, :] = (1.0 / scale) * traindata_label[
                                                                          idx_start + scale * crop_half: idx_start + scale * crop_half + scale * label_size:scale,
                                                                          idy_start + scale * crop_half: idy_start + scale * crop_half + scale * label_size:scale]

        traindata_batch_90d = np.float32((1 / 255) * traindata_batch_90d)
        traindata_batch_0d = np.float32((1 / 255) * traindata_batch_0d)
        traindata_batch_45d = np.float32((1 / 255) * traindata_batch_45d)
        traindata_batch_m45d = np.float32((1 / 255) * traindata_batch_m45d)

        traindata_batch_90d = np.expand_dims(np.minimum(np.maximum(traindata_batch_90d, 0), 1), axis=1)
        traindata_batch_0d = np.expand_dims(np.minimum(np.maximum(traindata_batch_0d, 0), 1), axis=1)
        traindata_batch_45d = np.expand_dims(np.minimum(np.maximum(traindata_batch_45d, 0), 1), axis=1)
        traindata_batch_m45d = np.expand_dims(np.minimum(np.maximum(traindata_batch_m45d, 0), 1), axis=1)

        traindata_batch = np.concatenate(
            [traindata_batch_0d, traindata_batch_45d, traindata_batch_90d, traindata_batch_m45d], axis=1)

        return traindata_batch, traindata_batch_label

    def choose_delta(self, image_id):
        # start = time.time()
        trans = np.random.randint(25)
        if trans < 15:  # 0
            traindata_all = self.traindata_all[image_id]
            traindata_label = self.traindata_label[image_id]
        elif trans < 18:
            # 偏移一位
            traindata_all = self.traindata_all_add1[image_id]
            traindata_label = self.traindata_label[image_id] + 1
        elif trans < 21:
            traindata_all = self.traindata_all_sub1[image_id]
            traindata_label = self.traindata_label[image_id] - 1
        elif trans < 23:
            # 偏移一位
            traindata_all = self.traindata_all_add2[image_id]
            traindata_label = self.traindata_label[image_id] + 2
        elif trans < 25:
            traindata_all = self.traindata_all_sub2[image_id]
            traindata_label = self.traindata_label[image_id] - 2

        pixel_random = np.random.randint(5)
        if pixel_random ==0:
            H,W,M,M,C = traindata_all.shape
            H,W = traindata_label.shape
            offsetx = (np.random.rand() - 0.5) *2
            offsety = (np.random.rand() - 0.5) *2
            mat_translation = np.float32([[1, 0, 1 * offsetx], [0, 1, 1 * offsety]])
            traindata_all= cv2.warpAffine( traindata_all.reshape(H,W,-1), mat_translation,(512, 512))
            traindata_label= cv2.warpAffine( traindata_label.reshape(H,W,-1), mat_translation,(512, 512))
            traindata_all = traindata_all.reshape(H,W,M,M,C)
            traindata_label = traindata_label.reshape(H,W)
        return traindata_all, traindata_label

    def data_aug(self, traindata_batch, traindata_label_batchNxN):
        """
             For Data augmentation
             (rotation, transpose and gamma noise)

         """

        traindata_batch_90d = traindata_batch[:, 2]
        traindata_batch_0d = traindata_batch[:, 0]
        traindata_batch_45d = traindata_batch[:, 1]
        traindata_batch_m45d = traindata_batch[:, 3]

        for batch_i in range(self.batch_size):
            gray_rand = 0.4 * np.random.rand() + 0.8
            traindata_batch_90d[batch_i, :, :, :] = pow(traindata_batch_90d[batch_i, :, :, :], gray_rand)
            traindata_batch_0d[batch_i, :, :, :] = pow(traindata_batch_0d[batch_i, :, :, :], gray_rand)
            traindata_batch_45d[batch_i, :, :, :] = pow(traindata_batch_45d[batch_i, :, :, :], gray_rand)
            traindata_batch_m45d[batch_i, :, :, :] = pow(traindata_batch_m45d[batch_i, :, :, :], gray_rand)


            transp_rand = np.random.randint(0, 2)

            if transp_rand == 1:  # 这个是转置
                traindata_batch_90d_tmp6 = np.copy(
                    np.transpose(np.squeeze(traindata_batch_90d[batch_i, :, :, :]), (1, 0, 2)))
                traindata_batch_0d_tmp6 = np.copy(
                    np.transpose(np.squeeze(traindata_batch_0d[batch_i, :, :, :]), (1, 0, 2)))
                traindata_batch_45d_tmp6 = np.copy(
                    np.transpose(np.squeeze(traindata_batch_45d[batch_i, :, :, :]), (1, 0, 2)))
                traindata_batch_m45d_tmp6 = np.copy(
                    np.transpose(np.squeeze(traindata_batch_m45d[batch_i, :, :, :]), (1, 0, 2)))

                traindata_batch_0d[batch_i, :, :, :] = np.copy(traindata_batch_90d_tmp6[:, :, ::-1])
                traindata_batch_90d[batch_i, :, :, :] = np.copy(traindata_batch_0d_tmp6[:, :, ::-1])
                traindata_batch_45d[batch_i, :, :, :] = np.copy(traindata_batch_45d_tmp6[:, :, ::-1])
                traindata_batch_m45d[batch_i, :, :, :] = np.copy(traindata_batch_m45d_tmp6)  # [:,:,::-1]) 正确
                traindata_label_batchNxN[batch_i, :, :] = np.copy(
                    np.transpose(traindata_label_batchNxN[batch_i, :, :], (1, 0)))


            rotation_or_transp_rand = np.random.randint(0, 4)
            if rotation_or_transp_rand == 1:  # 90도

                traindata_batch_90d_tmp3 = np.copy(np.rot90(traindata_batch_90d[batch_i, :, :, :], 1, (0, 1)))
                traindata_batch_0d_tmp3 = np.copy(np.rot90(traindata_batch_0d[batch_i, :, :, :], 1, (0, 1)))
                traindata_batch_45d_tmp3 = np.copy(np.rot90(traindata_batch_45d[batch_i, :, :, :], 1, (0, 1)))
                traindata_batch_m45d_tmp3 = np.copy(np.rot90(traindata_batch_m45d[batch_i, :, :, :], 1, (0, 1)))

                traindata_batch_90d[batch_i, :, :, :] = traindata_batch_0d_tmp3
                traindata_batch_45d[batch_i, :, :, :] = traindata_batch_m45d_tmp3
                traindata_batch_0d[batch_i, :, :, :] = traindata_batch_90d_tmp3[:, :, ::-1]
                traindata_batch_m45d[batch_i, :, :, :] = traindata_batch_45d_tmp3[:, :, ::-1]

                traindata_label_batchNxN[batch_i, :, :] = np.copy(
                    np.rot90(traindata_label_batchNxN[batch_i, :, :], 1, (0, 1)))



            if rotation_or_transp_rand == 2:  # 180도

                traindata_batch_90d_tmp4 = np.copy(np.rot90(traindata_batch_90d[batch_i, :, :, :], 2, (0, 1)))
                traindata_batch_0d_tmp4 = np.copy(np.rot90(traindata_batch_0d[batch_i, :, :, :], 2, (0, 1)))
                traindata_batch_45d_tmp4 = np.copy(np.rot90(traindata_batch_45d[batch_i, :, :, :], 2, (0, 1)))
                traindata_batch_m45d_tmp4 = np.copy(np.rot90(traindata_batch_m45d[batch_i, :, :, :], 2, (0, 1)))

                traindata_batch_90d[batch_i, :, :, :] = traindata_batch_90d_tmp4[:, :, ::-1]
                traindata_batch_0d[batch_i, :, :, :] = traindata_batch_0d_tmp4[:, :, ::-1]
                traindata_batch_45d[batch_i, :, :, :] = traindata_batch_45d_tmp4[:, :, ::-1]
                traindata_batch_m45d[batch_i, :, :, :] = traindata_batch_m45d_tmp4[:, :, ::-1]

                traindata_label_batchNxN[batch_i, :, :] = np.copy(
                    np.rot90(traindata_label_batchNxN[batch_i, :, :], 2, (0, 1)))



            if rotation_or_transp_rand == 3:  # 270도

                traindata_batch_90d_tmp5 = np.copy(np.rot90(traindata_batch_90d[batch_i, :, :, :], 3, (0, 1)))
                traindata_batch_0d_tmp5 = np.copy(np.rot90(traindata_batch_0d[batch_i, :, :, :], 3, (0, 1)))
                traindata_batch_45d_tmp5 = np.copy(np.rot90(traindata_batch_45d[batch_i, :, :, :], 3, (0, 1)))
                traindata_batch_m45d_tmp5 = np.copy(np.rot90(traindata_batch_m45d[batch_i, :, :, :], 3, (0, 1)))

                traindata_batch_90d[batch_i, :, :, :] = traindata_batch_0d_tmp5[:, :, ::-1]
                traindata_batch_0d[batch_i, :, :, :] = traindata_batch_90d_tmp5
                traindata_batch_45d[batch_i, :, :, :] = traindata_batch_m45d_tmp5[:, :, ::-1]
                traindata_batch_m45d[batch_i, :, :, :] = traindata_batch_45d_tmp5

                traindata_label_batchNxN[batch_i, :, :] = np.copy(
                    np.rot90(traindata_label_batchNxN[batch_i, :, :], 3, (0, 1)))


        traindata_batch_90d = np.expand_dims(traindata_batch_90d, axis=1)
        traindata_batch_0d = np.expand_dims(traindata_batch_0d, axis=1)
        traindata_batch_45d = np.expand_dims(traindata_batch_45d, axis=1)
        traindata_batch_m45d = np.expand_dims(traindata_batch_m45d, axis=1)

        traindata_batch = np.concatenate(
            [traindata_batch_0d, traindata_batch_45d, traindata_batch_90d, traindata_batch_m45d], axis=1)


        # noise添加
        mask = np.ones((traindata_batch.shape[0],traindata_batch.shape[2],traindata_batch.shape[3]))
        for batch_i in range(self.batch_size):
            # 高斯噪声
            traindata_batch[batch_i],mask[batch_i] = self.add_noise(traindata_batch[batch_i],mask[batch_i])
        mask = mask[:,1:-1,1:-1]
        # refine数据增强

        return traindata_batch, traindata_label_batchNxN,mask

    def transform(self, traindata_batch, traindata_label_batchNxN):  # 傅里叶变换

        pass

    def add_noise(self, traindata,mask):
        noise_rand = np.random.randint(0, 40)
        N,H,W,C = traindata.shape
        if noise_rand in [0, 1, 2, 3]:  # 高斯噪声
            # print(noise_rand)
            gauss = np.random.normal(0.0, np.random.uniform() * np.sqrt(0.2), (
                traindata.shape[0], traindata.shape[1], traindata.shape[2],
                traindata.shape[3]))
            traindata = np.clip(traindata + gauss, 0.0, 1.0)
        if noise_rand in [-1]:  # 椒盐噪声
            prob = 0.05
            rdn = np.random.rand(
                traindata.shape[0], traindata.shape[1], traindata.shape[2],
                traindata.shape[3])
            traindata[rdn < prob] = 0
            traindata[rdn > 1 - prob] = 1
        return traindata,mask

    def val_data(self, idx):
        batch_size = 1
        label_size, input_size = self.inputsize, self.inputsize

        test_data_90d = np.zeros((batch_size, input_size, input_size, self.views),
                                 dtype=np.float32)
        test_data_0d = np.zeros((batch_size, input_size, input_size, self.views),
                                dtype=np.float32)
        test_data_45d = np.zeros((batch_size, input_size, input_size, self.views),
                                 dtype=np.float32)
        test_data_m45d = np.zeros((batch_size, input_size, input_size, self.views),
                                  dtype=np.float32)

        test_data_label = np.zeros((batch_size, label_size, label_size))
        crop_half = int(0.5 * (input_size - label_size))

        R = 0.299
        G = 0.587
        B = 0.114

        ix_rd = 0
        iy_rd = 0
        start1 = 0
        end1 = self.views - 1

        test_image = self.traindata_all[idx]
        test_label = self.traindata_label[idx]


        seq0to8 = np.array([i for i in range(self.views)]) + ix_rd
        seq8to0 = np.array([i for i in range(self.views - 1, -1, -1)]) + iy_rd
        test_data_0d[0] = np.squeeze(
            R * test_image[:, :, 4 + ix_rd, seq0to8, 0].astype('float32') +
            G * test_image[:, :, 4 + ix_rd, seq0to8, 1].astype('float32') +
            B * test_image[:, :, 4 + ix_rd, seq0to8, 2].astype('float32'))

        test_data_90d[0] = np.squeeze(
            R * test_image[:, :, seq8to0, 4 + iy_rd, 0].astype('float32') +
            G * test_image[:, :, seq8to0, 4 + iy_rd, 1].astype('float32') +
            B * test_image[:, :, seq8to0, 4 + iy_rd, 2].astype('float32'))

        for kkk in range(start1, end1 + 1):
            test_data_45d[0, :, :, kkk - start1] = np.squeeze(
                R * test_image[:, :, (8) - kkk + ix_rd, kkk + iy_rd, 0].astype('float32') +
                G * test_image[:, :, (8) - kkk + ix_rd, kkk + iy_rd, 1].astype('float32') +
                B * test_image[:, :, (8) - kkk + ix_rd, kkk + iy_rd, 2].astype('float32'))

            test_data_m45d[0, :, :, kkk - start1] = np.squeeze(
                R * test_image[:, :, kkk + ix_rd, kkk + iy_rd, 0].astype('float32') +
                G * test_image[:, :, kkk + ix_rd, kkk + iy_rd, 1].astype('float32') +
                B * test_image[:, :, kkk + ix_rd, kkk + iy_rd, 2].astype('float32'))


        test_data_label[0] = test_label[crop_half: crop_half + label_size,
                             crop_half:crop_half + label_size]

        test_data_90d = np.float32((1 / 255) * test_data_90d)
        test_data_0d = np.float32((1 / 255) * test_data_0d)
        test_data_45d = np.float32((1 / 255) * test_data_45d)
        test_data_m45d = np.float32((1 / 255) * test_data_m45d)

        test_data_90d = np.expand_dims(np.minimum(np.maximum(test_data_90d, 0), 1), axis=1)
        test_data_0d = np.expand_dims(np.minimum(np.maximum(test_data_0d, 0), 1), axis=1)
        test_data_45d = np.expand_dims(np.minimum(np.maximum(test_data_45d, 0), 1), axis=1)
        test_data_m45d = np.expand_dims(np.minimum(np.maximum(test_data_m45d, 0), 1), axis=1)

        test_data = np.concatenate(
            [test_data_0d, test_data_45d, test_data_90d, test_data_m45d], axis=1)
        return test_data, test_data_label