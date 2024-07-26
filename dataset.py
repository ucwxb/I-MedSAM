import os
join = os.path.join

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import albumentations as A
import math
import random

def make_odd(num):
    num = math.ceil(num)
    if num % 2 == 0:
        num += 1
    return num

def apply_transform(image, mask):
    strategy = [(1, 2), (0, 3), (0, 2), (1, 1)]
    level = 5
    transform = A.Compose([
        A.ColorJitter(brightness=0.04 * level, contrast=0, saturation=0, hue=0, p=0.2 * level),
        A.ColorJitter(brightness=0, contrast=0.04 * level, saturation=0, hue=0, p=0.2 * level),
        A.Posterize(num_bits=math.floor(8 - 0.8 * level), p=0.2 * level),
        A.Sharpen(alpha=(0.04 * level, 0.1 * level), lightness=(1, 1), p=0.2 * level),
        A.GaussianBlur(blur_limit=(3, make_odd(3 + 0.8 * level)), p=0.2 * level),
        A.GaussNoise(var_limit=(2 * level, 10 * level), mean=0, per_channel=True, p=0.2 * level),
        A.Rotate(limit=4 * level, interpolation=1, border_mode=0, value=0, mask_value=None, rotate_method='largest_box',
                    crop_border=False, p=0.2 * level),
        A.HorizontalFlip(p=0.2 * level),
        A.VerticalFlip(p=0.2 * level),
        A.Affine(scale=(1 - 0.04 * level, 1 + 0.04 * level), translate_percent=None, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                    shear={'x': (0, 2 * level), 'y': (0, 0)}
                    , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),  # x
        A.Affine(scale=None, translate_percent=None, translate_px=None, rotate=None,
                    shear={'x': (0, 0), 'y': (0, 2 * level)}
                    , interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0.02 * level), 'y': (0, 0)}, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level),
        A.Affine(scale=None, translate_percent={'x': (0, 0), 'y': (0, 0.02 * level)}, translate_px=None, rotate=None,
                    shear=None, interpolation=1, mask_interpolation=0, cval=0, cval_mask=0, mode=0, fit_output=False,
                    keep_ratio=True, p=0.2 * level)
    ])
    employ = random.choice(strategy)
    level, shape = random.sample(transform[:6], employ[0]), random.sample(transform[6:], employ[1])
    img_transform = A.Compose([*level, *shape])
    random.shuffle(img_transform.transforms)
    transformed = img_transform(image=image, mask=mask)
    return transformed['image'], transformed['mask']

class SessileDataLoader(Dataset):
    def __init__(self, data_root, train=True, resize_size=[384, 384], label_resize_size=[], train_ratio=1.0):
        self.train = train
        self.data_root = data_root
        self.resize_size = resize_size
        if len(label_resize_size) <= 0:
            self.label_resize_size = resize_size
        else:
            self.label_resize_size = label_resize_size

        self.img_files = []
        self.gt_files = []
        self.class_num=1
        self.image_size=resize_size[0]

        if train:
            self.scan_list = os.listdir(os.path.join(self.data_root,"train", "images"))
            self.scan_list.sort()
            self.scan_list = self.scan_list[:int(len(self.scan_list) * train_ratio)]
            for scan in self.scan_list:
                self.img_files.append(os.path.join(self.data_root,"train", "images", scan))
                self.gt_files.append(os.path.join(self.data_root,"train", "masks", scan))
        else:
            self.scan_list = os.listdir(os.path.join(self.data_root,"test", "images"))
            self.scan_list.sort()
            for scan in self.scan_list:
                self.img_files.append(os.path.join(self.data_root,"test", "images", scan))
                self.gt_files.append(os.path.join(self.data_root,"test", "masks", scan))
        
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = cv2.imread(self.img_files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.resize_size)

        gt2D = cv2.imread(self.gt_files[index], cv2.IMREAD_GRAYSCALE)
        gt2D = cv2.resize(gt2D, self.label_resize_size, cv2.INTER_NEAREST)

        if self.train:
            while True:
                img, gt2D = apply_transform(img, gt2D)
                if np.sum(gt2D) > 0:
                    break

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        gt2D = gt2D.astype(np.uint8) / 255
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        if self.train:
            H, W = gt2D.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        box = np.array([x_min, y_min, x_max, y_max])
        return torch.tensor(img).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(box).float(), torch.tensor(0).long(), self.img_files[index], self.gt_files[index]

class CVCDataLoader(Dataset):
    def __init__(self, data_root, train=True, resize_size=[384, 384], label_resize_size=[]):
        self.train = train
        self.data_root = os.path.join(data_root, "PNG")
        self.resize_size = resize_size
        if len(label_resize_size) <= 0:
            self.label_resize_size = resize_size
        else:
            self.label_resize_size = label_resize_size
        self.img_files = []
        self.gt_files = []
        self.class_num=1
        self.image_size=resize_size[0]

        scan_list = os.listdir(os.path.join(self.data_root, "Original"))
        scan_list.sort()
        if train:
            scan_list = scan_list[:]
        else:
            scan_list = scan_list[:]
        
        for scan in scan_list:
            self.img_files.append(os.path.join(self.data_root, "Original", scan))
            self.gt_files.append(os.path.join(self.data_root, "Ground Truth", scan))
    
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img = cv2.imread(self.img_files[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.resize_size)

        gt2D = cv2.imread(self.gt_files[index], cv2.IMREAD_GRAYSCALE)
        gt2D = cv2.resize(gt2D, self.label_resize_size, cv2.INTER_NEAREST)

        if self.train:
            while True:
                img, gt2D = apply_transform(img, gt2D)
                if np.sum(gt2D) > 0:
                    break

        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)

        gt2D = gt2D.astype(np.uint8) / 255
        
        y_indices, x_indices = np.where(gt2D > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        if self.train:
            H, W = gt2D.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        box = np.array([x_min, y_min, x_max, y_max])
        
        return torch.tensor(img).float(), torch.tensor(gt2D[None, :,:]).long(), torch.tensor(box).float(), torch.tensor(0).long(), self.img_files[index], self.gt_files[index]
