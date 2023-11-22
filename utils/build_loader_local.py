import copy
import math
import os
import random
from random import randint

import albumentations
import cv2
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as tf
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb


def build_loader_local(config):
    dataset_root = config.DATA.DATA_PATH
    traintxt = config.DATA.TRAIN_LIST
    valtxt = config.DATA.VAL_LIST
    testtxt = config.DATA.TEST_LIST

    train_list = []
    train_label = []
    val_list = []
    val_label = []
    test_list = []
    test_label = []

    with open(traintxt, encoding='utf-8') as e: # load train list and train label
        list = e.readlines()
        for i in list:
            train_list.append(i.split(' ')[0])
            # label =  i.split(' ')[1:15]
            label = [int(x) for x in i.split(' ')[1:15]]
            train_label.append(label)
    with open(valtxt, encoding='utf-8') as e: # load train list and train label
        list = e.readlines()
        for i in list:
            val_list.append(i.split(' ')[0])
            label = [int(x) for x in i.split(' ')[1:15]]
            val_label.append(label)

    # if config.POPAR_FORM:
    train_list = np.hstack((train_list, val_list))
    train_label = np.vstack((train_label, val_label))

    train_dataset = NIHchest_dataset(dataset_root=dataset_root, datalist=train_list, config=config,
                                    img_transforms=img_transforms(config),
                                    patch_transforms=patch_transforms(mode='train'))
    print("successfully build train dataset")


    train_loader = DataLoader(dataset=train_dataset, 
                                # sampler=sampler_train,
                                batch_size=config.DATA.BATCH_SIZE, 
                                shuffle=True, 
                                num_workers=config.DATA.NUM_WORKERS,
                                drop_last=True)

    with open(testtxt, encoding='utf-8') as e: # load train list and train label
        list = e.readlines()
        for i in list:
            test_list.append(i.split(' ')[0])
            label = [int(x) for x in i.split(' ')[1:15]]
            test_label.append(label)

    val_dataset = NIHchest_dataset(dataset_root=dataset_root, datalist=test_list, config=config,
                                    img_transforms=img_transforms(config=config), 
                                    patch_transforms=patch_transforms(mode='val'))
    print("successfully build val dataset")
    # sampler_val = torch.utils.data.distributed.DistributedSampler(
    #     val_dataset, shuffle=config.TEST.SHUFFLE
    # )
    val_loader = DataLoader(dataset=val_dataset, 
                            # sampler=sampler_val,
                            batch_size=config.DATA.BATCH_SIZE, 
                            num_workers=config.DATA.NUM_WORKERS)

    return train_dataset, val_dataset, train_loader, val_loader

    

class NIHchest_dataset(Dataset):
    def __init__(self, dataset_root, datalist, config, img_transforms, patch_transforms):
        # super(NIHchest_dataset, self).__init__()
        self.img_transforms = img_transforms
        self.patch_transforms = patch_transforms
        self.dataset_root = dataset_root
        self.datalist = datalist
        self.image_size = config.DATA.IMG_SIZE

    def __getitem__(self, index):

        image = cv2.imread(os.path.join(self.dataset_root, self.datalist[index]))

        # global embedding consistency data
        image = self.img_transforms(image)
        patch1, patch2, (x1, y1), (x2, y2) = patch_crop(image)
        index_l_1, index_l_2, ratio_l, index_s_1, index_s_2, ratio_s = get_index((x1, y1), (x2, y2))

        patch1 = self.patch_transforms(patch1)
        patch2 = self.patch_transforms(patch2)
   

        return patch1.float(), patch2.float(), index_l_1, index_l_2, ratio_l, index_s_1, index_s_2, ratio_s
    
    def __len__(self):
        return len(self.datalist)


def patch_crop(image):
    x1 = randint(0, 63)
    y1 = randint(0, 63)
    patch1 = image[x1:x1+448, y1:y1+448, :]

    x2 = randint(0,287)
    y2 = randint(0,287)
    patch2 = image[x2:x2+224, y2:y2+224, :]
    patch2 = cv2.resize(patch2, (448, 448), interpolation=cv2.INTER_AREA)
    
    return patch1, patch2, (x1, y1), (x2, y2)

def get_index(a, b): # a为大patch左上角坐标，b为小patch左上角坐标
    (x1, y1), (x2, y2) = a, b

    # 重合部分坐标范围
    xa, xb = max(x1, x2), min((x1+448), (x2+224))
    ya, yb = max(y1, y2), min((y1+448), (y2+224))

    # 对于大crop：crop_l，计算横纵索引的取值范围以及重合patch的索引
    idx_cropl_x1, idx_cropl_x2 = math.ceil((xa-x1)/32), int((xb-x1)/32)-1
    idx_cropl_y1, idx_cropl_y2 = math.ceil((ya-y1)/32), int((yb-y1)/32)-1
    index_l = []
    for i in range(idx_cropl_y1, idx_cropl_y2+1):
        for j in range(idx_cropl_x1, idx_cropl_x2+1):
            idx = i*14+j
            index_l.append(idx)

    # 对于小crop：crop_s，计算横纵索引的取值范围以及重合patch的索引
    idx_crops_x1, idx_crops_x2 = math.ceil((xa-x2)/16), int((xb-x2)/16)-1
    idx_crops_y1, idx_crops_y2 = math.ceil((ya-y2)/16), int((yb-y2)/16)-1
    index_s = []
    for i in range(idx_crops_y1, idx_crops_y2+1):
        for j in range(idx_crops_x1, idx_crops_x2+1):
            idx = i*14+j
            index_s.append(idx)
    
    # 对于大crop重合索引，随机取出一半，并找出小crop对应的索引
    index_l_1 = random.sample(index_l, int(len(index_l)/2))
    index_l_2 = [] # 对应的小crop的patch的索引
    ratio_l = [] # patch之间的重合率
    for i in index_l_1:
        center_x, center_y = (i%14)*32+16+x1, int(i/14)*32+16+y1 # 大crop的patch的中心点坐标
        index_x, index_y = int((center_x-x2)/16), int((center_y-y2)/16)
        index = index_y*14+index_x # 对应的小crop的patch的索引
        index_l_2.append(index)

        center_x_2, center_y_2 = (index%14)*16+8+x2, int(index/14)*16+8+y2 # 对应的小crop的patch的中心点坐标
        
        x_hat = min((center_x_2+8), (center_x+16)) - max((center_x-16), (center_x_2-8)) # 重合部分的长宽
        y_hat = min((center_y_2+8), center_y+16) - max((center_y-16), (center_y_2-8))

        ratio = (x_hat*y_hat*1.0)/(32*32)
        ratio_l.append(ratio)

    # 对于小crop重合索引，随机取出1/4，并找出大crop对应的索引
    index_s_1 = random.sample(index_s, int(len(index_s)/4))
    index_s_2 = [] # 对应的大crop的patch的索引
    ratio_s = []
    for i in index_s_1:
        center_x, center_y = (i%14)*16+8+x2, int(i/14)*16+8+y2 # 小crop的patch的中心点坐标
        index_x, index_y = int((center_x-x1)/32), int((center_y-y1)/32)
        index = index_y*14+index_x # 对应大crop的patch的索引
        index_s_2.append(index)

        center_x_2, center_y_2 = (index%14)*32+16+x1, int(index/14)*32+16+y1 # 对应大crop的patch的中心点坐标
        
        x_hat = min((center_x_2+16), (center_x+8)) - max((center_x_2-16), center_x-8)
        y_hat = min((center_y_2+16), (center_y+8)) - max((center_y_2-16), center_y-8)
        
        ratio = (x_hat*y_hat*1.0)/(32*32)
        ratio_s.append(ratio)
    
    for l in index_l_1, index_l_2, ratio_l, index_s_1, index_s_2, ratio_s:
        if len(l)<196:
            for i in range(len(l), 196):
                l.append(1000)
    index_l_1, index_l_2, ratio_l, index_s_1, index_s_2, ratio_s = torch.tensor(index_l_1), torch.tensor(index_l_2), torch.tensor(ratio_l), torch.tensor(index_s_1), torch.tensor(index_s_2), torch.tensor(ratio_s)
    return index_l_1, index_l_2, ratio_l, index_s_1, index_s_2, ratio_s
    



def img_transforms(config):
    if config.DATA.IMG_SIZE == 448:
        size = 512
    elif config.DATA.IMG_SIZE == 224:
        size = 256
    img_transforms = transforms.Compose([
                        KeepRatioResize(size),
                        # CenterCrop(size),
                        # GridRandomCrop(size),
                        # PatchCrop(size)
    ])
    return img_transforms

def patch_transforms(mode):
    if mode == 'train':
        img_transforms = transforms.Compose([
                            To_Tensor(),
                            # transforms.GaussianBlur(kernel_size=5),
                            # transforms.RandomHorizontalFlip(p=0.5),
                            # transforms.RandomRotation(degrees=7),
                            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
        ])
    elif mode == 'val':
        img_transforms = transforms.Compose([
                            To_Tensor(),
                            transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
        ])
    return img_transforms

def build_md_transform(mode, dataset = "chexray"):
    transformList_mg = []
    transformList_simple = []

    if dataset == "imagenet":
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])


    if mode=="train":
        transformList_mg.append(Rearrange_and_Norm())
        transformList_mg.append(local_pixel_shuffling)
        transformList_mg.append(nonlinear_transformation)
        transformList_mg.append(transforms.RandomApply([paint], p=0.9))
        transformList_mg.append(torch.from_numpy)
        transformList_mg.append(normalize)
        transformSequence_mg = transforms.Compose(transformList_mg)

        transformList_simple.append(Rearrange_and_Norm())
        transformList_simple.append(torch.from_numpy)
        transformList_simple.append(normalize)
        transformSequence_simple = transforms.Compose(transformList_simple)

        return transformSequence_mg, transformSequence_simple
    else:
        transformList_simple.append(Rearrange_and_Norm())
        transformList_simple.append(torch.from_numpy)
        transformList_simple.append(normalize)
        transformSequence_simple = transforms.Compose(transformList_simple)
        return transformSequence_simple, transformSequence_simple

class Rearrange_and_Norm():
    def __call__(self, image):
        # image = cv2.resize(image, (self.size, self.size))
        image = rearrange(image, 'h w c-> c h w')/255
        return image

class To_Tensor():
    def __call__(self, image):
        # if len(np.array(image).shape) == 2:
        #     image = np.array(image)[:,:,None]
        image = rearrange(image, 'h w c-> c h w')/255
        image = torch.from_numpy(image)
        return image

class KeepRatioResize(): # keep ratio and resize: input size -> h*608 / 608*w
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        h,w,c = image.shape
        if h>w:
            new_h = int(self.size*h*1.0/w)
            new_w = self.size
        else:
            new_h = self.size
            new_w = int(self.size*w*1.0/h)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return image




class CenterCrop(): # h*608 / 608*w -> 608*608
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        image = rearrange(image, 'h w c-> c h w')
        image = torch.from_numpy(image)
        image = tf.center_crop(image, [self.size,self.size])
        image = image.numpy()
        image = rearrange(image, 'c h w-> h w c')
        return image

class GridRandomCrop(): # 608*608 -> 576*576 / 304*304 -> 288*288
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        grid_size = int(self.size/19) # the image is cropped into 19*19 grids, and we take 18*18 grids from it
        start_x = randint(0,grid_size-1)
        start_y = randint(0,grid_size-1)
        image = image[start_x:grid_size*18+start_x, start_y:grid_size*18+start_y, :]
        return image

class PatchCrop():
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        grid = int(self.size/19)
        x1 = randint(0,3)
        y1 = randint(0,3)
        x2 = randint(0,3)
        y2 = randint(0,3)
        patch1 = image[x1*grid:(14+x1)*grid, y1*grid:(14+y1)*grid, :]
        patch2 = image[x2*grid:(14+x2)*grid, y2*grid:(14+y2)*grid, :]
        image = np.concatenate((patch1, patch2), axis=2) # 448*448*6
        return image

def local_pixel_shuffling(x, prob=0.5):

    # if random.random() >= prob:
    #     return x


    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    img_deps, img_rows, img_cols = x.shape
    num_block = 40
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows // 10)
        block_noise_size_y = random.randint(1, img_cols // 10)
        noise_x = random.randint(0, img_rows - block_noise_size_x)
        noise_y = random.randint(0, img_cols - block_noise_size_y)
        window = orig_image[:, noise_x:noise_x + block_noise_size_x, noise_y:noise_y + block_noise_size_y]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((img_deps, block_noise_size_x,
                                 block_noise_size_y))
        image_temp[:, noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = window
    local_shuffling_x = image_temp


    return local_shuffling_x

def nonlinear_transformation(x, prob=0.9):


    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=500)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)

    return nonlinear_x


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def paint(img,outpaint_rate = 0.8):


    if random.random() < outpaint_rate:
        return image_out_painting(img)
    else:
        return image_in_painting(img)

def image_in_painting(x):
    img_deps, img_rows, img_cols = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows // 6, img_rows // 3)
        block_noise_size_y = random.randint(img_cols // 6, img_cols // 3)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)

        noise = np.random.rand(block_noise_size_x,block_noise_size_y) * 1.0

        for i in range(0,img_deps):
            x[i, noise_x:noise_x + block_noise_size_x,noise_y:noise_y + block_noise_size_y] = noise
        cnt -= 1


    return x


def image_out_painting(x):

    img_deps, img_rows, img_cols = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2] ) * 1.0
    block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
    block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
    noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
    noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
    x[:, noise_x:noise_x + block_noise_size_x,
    noise_y:noise_y + block_noise_size_y] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                            noise_y:noise_y + block_noise_size_y]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        noise_x = random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = random.randint(3, img_cols - block_noise_size_y - 3)
        x[:, noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                noise_y:noise_y + block_noise_size_y]
        cnt -= 1


    return x