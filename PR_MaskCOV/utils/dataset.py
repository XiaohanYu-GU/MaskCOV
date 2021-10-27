# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import pandas
import random
import PIL.Image as Image
from PIL import ImageStat
import numpy as np

def random_sample(img_names, labels):
    anno_dict = {}
    img_list = []
    anno_list = []
    for img, anno in zip(img_names, labels):
        if not anno in anno_dict:
            anno_dict[anno] = [img]
        else:
            anno_dict[anno].append(img)

    for anno in anno_dict.keys():
        anno_len = len(anno_dict[anno])
        fetch_keys = random.sample(list(range(anno_len)), anno_len//10)
        img_list.extend([anno_dict[anno][x] for x in fetch_keys])
        anno_list.extend([anno for x in fetch_keys])
    return img_list, anno_list

class dataset(data.Dataset):
    def __init__(self, Config, anno, swap_size=[2,2], common_aug=None, swap=None, totensor=None, train=False, train_val=False, test=False):
        self.Config = Config
        self.root_path = Config.rawdata_root
        self.numcls = Config.numcls
        self.dataset = Config.dataset
        self.use_cls_2 = Config.cls_2
        self.use_cls_mul = Config.cls_2xmul
        if isinstance(anno, pandas.core.frame.DataFrame):
            self.paths = anno['ImageName'].tolist()
            self.labels = anno['label'].tolist()
        elif isinstance(anno, dict):
            self.paths = anno['img_name']
            self.labels = anno['label']

        if train_val:
            self.paths, self.labels = random_sample(self.paths, self.labels)
        self.common_aug = common_aug
        self.swap = swap
        self.totensor = totensor
        self.cfg = Config
        self.train = train
        self.swap_size = swap_size
        self.test = test

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])

        img = self.pil_loader(img_path)

        if self.test:
            img = self.totensor(img)
            label = self.labels[item] - 1
            return img, label, self.paths[item]

        if self.swap_size[1] == 2:
            foo = range(4)
        elif self.swap_size[1] == 3:
            foo = range(9)
        elif self.swap_size[1] == 4:
            foo = range(16)
        elif self.swap_size[1] == 6:
            foo = range(36)

        mask_num = random.sample(foo, self.Config.mask_num)  # masked number
        j = mask_num[0] % self.swap_size[0]
        i = mask_num[0] // self.swap_size[0]
        h_space = int(384/self.swap_size[0])
        w_space = int(384/self.swap_size[0])

        mask = np.ones([384, 384])
        mask[i * h_space: (i + 1) * h_space, j * w_space: (j + 1) * w_space] = 0
        mask = mask.reshape(384, 384, 1)

        # # custom fine-grained dataset input = 448
        # mask_num = random.sample(foo, 1)  # 遮住几块
        # j = mask_num[0] % self.swap_size[0]
        # i = mask_num[0] // self.swap_size[0]
        # h_space = int(448/self.swap_size[0])
        # w_space = int(448/self.swap_size[0])

        # mask = np.ones([448, 448])
        # mask[i * h_space: (i + 1) * h_space, j * w_space: (j + 1) * w_space] = 0
        # mask = mask.reshape(448, 448, 1)
        # custom fine-grained dataset

        img_unswap = self.common_aug(img) if not self.common_aug is None else img
        image_unswap_list_original = self.crop_image(img_unswap, self.swap_size)  # frojm left to right, from top to bottom

        covaMatrix_list_unswap_original = []  # original img cova
        for crop_img in image_unswap_list_original:
            covariance_matrix = self.cal_covariance(crop_img)
            covaMatrix_list_unswap_original.append(covariance_matrix)

        img_unswap_mask = np.array(img_unswap) * mask
        img_unswap_mask = Image.fromarray(np.uint8(img_unswap_mask))

        image_unswap_list = self.crop_image(img_unswap_mask, self.swap_size)

        covaMatrix_list_unswap = []  # original img with mask cova
        for crop_img in image_unswap_list:
            covariance_matrix = self.cal_covariance(crop_img)
            covaMatrix_list_unswap.append(covariance_matrix)

        if self.train:
            img_swap = self.swap(img_unswap)  # Randomswap
            image_swap_list = self.crop_image(img_swap, self.swap_size)

            unswap_stats = [sum(ImageStat.Stat(im).mean) for im in image_unswap_list_original]
            swap_stats = [sum(ImageStat.Stat(im).mean) for im in image_swap_list]

            covaMatrix_list_swap = []
            for swap_im in swap_stats:
                distance = [abs(swap_im - unswap_im) for unswap_im in unswap_stats]
                index = distance.index(min(distance))
                covaMatrix_list_swap.append(covaMatrix_list_unswap_original[index])

            img_swap = np.array(img_swap) * mask
            img_swap = Image.fromarray(np.uint8(img_swap))
            for mask_index in mask_num:
                covaMatrix_list_swap[mask_index] = np.zeros((3, 3))

            covaMatrix_list_unswap_original = np.array(covaMatrix_list_unswap_original).reshape(-1).tolist()
            covaMatrix_list_swap = np.array(covaMatrix_list_swap).reshape(-1).tolist()
            covaMatrix_list_unswap = np.array(covaMatrix_list_unswap).reshape(-1).tolist()

            img_swap = self.totensor(img_swap)
            label = self.labels[item] - 1
            if self.use_cls_mul:
                label_swap = label + self.numcls
            if self.use_cls_2:
                label_swap = -1
            img_unswap = self.totensor(img_unswap)
            img_unswap_mask = self.totensor(img_unswap_mask)

            return img_unswap, img_unswap_mask, img_swap, label, label, label_swap, covaMatrix_list_unswap_original, covaMatrix_list_unswap, covaMatrix_list_swap, \
                   self.paths[item]

    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def crop_image(self, image, cropnum):
        width, high = image.size
        crop_x = [int((width / cropnum[0]) * i) for i in range(cropnum[0] + 1)]
        crop_y = [int((high / cropnum[1]) * i) for i in range(cropnum[1] + 1)]
        im_list = []
        for j in range(len(crop_y) - 1):
            for i in range(len(crop_x) - 1):
                im_list.append(image.crop((crop_x[i], crop_y[j], min(crop_x[i + 1], width), min(crop_y[j + 1], high))))
        return im_list


    def get_weighted_sampler(self):
        img_nums = len(self.labels)
        weights = [self.labels.count(x) for x in range(self.numcls)]
        return torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples=img_nums)

    def cal_covariance(self, input):
        img = np.array(input,np.float32)/255  # h w c
        h,w,c = img.shape
        img = img.transpose((2, 0, 1))
        img = img.reshape((3, -1))
        mean = img.mean(1)
        img = img -mean.reshape(3,1)

        covariance_matrix = np.matmul(img,np.transpose(img))
        covariance_matrix = covariance_matrix/(h*w-1)
        return covariance_matrix
        
# original, original mask, swap mask : 110
def collate_fn4train(batch):
    imgs = []
    label = []
    label_swap = []
    law_swap = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        imgs.append(sample[1])
        imgs.append(sample[2])
        label.append(sample[3])
        label.append(sample[3])
        label.append(sample[3])
        if sample[5] == -1:  # 110
            label_swap.append(1)
            label_swap.append(1)
            label_swap.append(0)
        else:
            label_swap.append(sample[2])
            label_swap.append(sample[3])
        law_swap.append(sample[6])
        law_swap.append(sample[7])
        law_swap.append(sample[8])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, label_swap, law_swap, img_name

def collate_fn4val(batch):
    imgs = []
    label = []
    label_swap = []
    law_swap = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        if sample[3] == -1:
            label_swap.append(1)
        else:
            label_swap.append(sample[2])
        law_swap.append(sample[3])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, label_swap, law_swap, img_name

def collate_fn4backbone(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        if len(sample) == 5:
            label.append(sample[1])
        else:
            label.append(sample[2])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name


def collate_fn4test(batch):
    imgs = []
    label = []
    img_name = []
    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])
        img_name.append(sample[-1])
    return torch.stack(imgs, 0), label, img_name
