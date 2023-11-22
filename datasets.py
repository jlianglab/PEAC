

import copy
import csv
import os
import random
import re
import sys
import time
from os.path import exists, isfile, join

import cv2
import numpy as np
import torch.utils.data
from einops import rearrange
from md_aug import (local_pixel_shuffling, local_pixel_shuffling_500,
                    nonlinear_transformation, paint)
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MaskGenerator:
    def __init__(self, input_size=448, mask_patch_size=32, model_patch_size=4, mask_ratio=0.5):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask


def build_md_transform(mode, dataset = "chexray"):
    transformList_mg = []
    transformList_simple = []

    if dataset == "imagenet":
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])


    if mode=="train":
        transformList_mg.append(local_pixel_shuffling)
        transformList_mg.append(nonlinear_transformation)
        transformList_mg.append(transforms.RandomApply([paint], p=0.9))
        transformList_mg.append(torch.from_numpy)
        transformList_mg.append(normalize)
        transformSequence_mg = transforms.Compose(transformList_mg)

        transformList_simple.append(torch.from_numpy)
        transformList_simple.append(normalize)
        transformSequence_simple = transforms.Compose(transformList_simple)

        return transformSequence_mg, transformSequence_simple
    else:
        transformList_simple.append(torch.from_numpy)
        transformList_simple.append(normalize)
        transformSequence_simple = transforms.Compose(transformList_simple)
        return transformSequence_simple, transformSequence_simple



class Popar_chestxray(Dataset):
    def __init__(self, image_path_file, augment, image_size=448,patch_size=32, ablation_mode='odadocar'):
        self.img_list = []
        self.augment = augment
        self.patch_size = patch_size
        self.image_size = image_size
        self.graycodes = []
        self.ablation_mode = ablation_mode
        self.mask_generator = MaskGenerator()

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = True
                while line:
                    line = fileDescriptor.readline().strip()
                    if line:
                        lineItems = line.split(" ")
                        imagePath = os.path.join(pathImageDirectory, lineItems[0])
                        self.img_list.append(imagePath)


    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),(self.image_size,self.image_size), interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        gt_whole = self.augment[1](imageData).float()
        aug_whole = self.augment[0](imageData).float()
        mask = self.mask_generator()
        order_patch = torch.arange(0,(self.image_size//self.patch_size)**2, dtype=torch.long)
        random_patch = torch.randperm((self.image_size//self.patch_size)**2, dtype=torch.long)


        if self.ablation_mode=='odadocar':
            if random.random()<0.5:
                randperm = order_patch
                aug_whole = aug_whole
            else:
                aug_whole = gt_whole
                randperm = random_patch
        elif self.ablation_mode=='odoc' or self.ablation_mode=='odar' or self.ablation_mode=='odocar':
            randperm = random_patch
            aug_whole = gt_whole
        elif self.ablation_mode=='adoc' or self.ablation_mode=='adar' or self.ablation_mode=='adocar':
            randperm = order_patch
            aug_whole = aug_whole
        elif self.ablation_mode=='ad2ar2':
            randperm = order_patch
            aug_whole = (gt_whole, mask)

        return randperm, gt_whole, aug_whole

    def __len__(self):
        return len(self.img_list)



class ChestX_ray14Xpert_MG_SEP(Dataset):
    def __init__(self, image_path_file, augment, shuffle_randomness=1,image_size=224,patch_size=16 ):
        self.img_list = []
        self.augment = augment
        self.shuffle_randomness = shuffle_randomness
        self.patch_size = patch_size
        self.image_size = image_size
        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = True
                while line:
                    line = fileDescriptor.readline()
                    if line:
                        lineItems = line.split()
                        imagePath = os.path.join(pathImageDirectory, lineItems[0])
                        self.img_list.append(imagePath)

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),(self.image_size,self.image_size), interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        gt_whole = self.augment[1](imageData)
        if random.random()<0.5:
            randperm = torch.arange(0,(self.image_size//self.patch_size)**2, dtype=torch.long)
            aug_whole = self.augment[0](imageData)
        else:
            aug_whole = gt_whole
            if self.shuffle_randomness == 1:
                randperm = torch.randperm((self.image_size//self.patch_size)**2)
            else:
                randperm = torch.arange(0, (self.image_size//self.patch_size)**2-1, dtype=torch.long)
                for i, p in enumerate(randperm):
                    if random.random() <= self.shuffle_randomness:
                        random_idx = random.randint(0, (self.image_size//self.patch_size)**2-1)
                        temp = randperm[i]
                        randperm[i] = randperm[random_idx]
                        randperm[random_idx] = temp
        return randperm, gt_whole.float(), aug_whole.float()

    def __len__(self):
        return len(self.img_list)