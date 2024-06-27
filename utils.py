import numpy as np
import random
import os
import argparse
from os.path import join

import torch
from torch import nn
from torch.utils.data.dataset import Dataset
from PIL import Image

import zipfile 
import tarfile
import pickle
  
def unzip_file(filepath, dest_path=None):
    
    file_type = filepath.split('.')[-1]
    if dest_path is None:
        dest_path = filepath.split('.')[0]
    print('uzip ing ..............')
    if file_type == 'zip':
        with zipfile.ZipFile(filepath, 'r') as zip_ref:  
            zip_ref.extractall(dest_path)  
    elif file_type in ['gz', 'tgz']:
        with tarfile.open(filepath, 'r:gz') as tar:  
            tar.extractall(path=dest_path)
    elif file_type == 'tar':
        with tarfile.open(filepath, 'r') as tar:  
            tar.extractall(path=dest_path)
            
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def make_arg():
    parser = argparse.ArgumentParser(description="SupervisedVSNot")
    parser.add_argument(
        'dataset',
        type=str
    )
    parser.add_argument(
        '--train-style', 
        default='Nonsupervied',
        help='Supervised or Nonsupervised')
    parser.add_argument(
        '--mode',
        default='train',
        type=str,
        help='train or test'
    )
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='LCP test model'
    )
    parser.add_argument(
        '--weight',
        default=None,
        type=str,
        help='LCP test weight'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        type=str,
        help='device of training'
    )
    parser.add_argument(
        '-b',
        '--batch-size',
        default=32,
        type=int,
        metavar="N",
        help='Batch size'
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        '-e',
        '--epochs',
        default=100,
        type=int,
        metavar='N',
        help="number of total epochs to run"
    )
    parser.add_argument(
        '--lr',
        '--learning-rate',
        default=0.003,
        type=float,
        metavar='LR',
        help="initial learning rate",
        dest='lr'
    )
    parser.add_argument(
        "--momentum", 
        default=0.9, 
        type=float, 
        metavar="M", 
        help="momentum of SGD solver"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--val-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        '-o',
        '--out-dir',
        default='./saves',
        type=str,
        metavar='DIR',
        help='the path of result save'
    )
    
    # arg of MoCo
    parser.add_argument(
        "--moco-dim", 
        default=128, 
        type=int, 
        help="feature dimension (default: 128)"
    )
    parser.add_argument(
        "--moco-k",
        default=65536,
        type=int,
        help="queue size; number of negative keys (default: 8192)",
    )
    parser.add_argument(
        "--moco-m",
        default=0.999,
        type=float,
        help="moco momentum of updating key encoder (default: 0.999)",
    )
    parser.add_argument(
        "--moco-t", 
        default=0.07, 
        type=float, 
        help="softmax temperature (default: 0.07)"
    )
    
    return parser

def get_resnet_weight(moco_weight):
    resnet_weight = dict()
    for key in moco_weight:
        if key.split('.')[0] == 'encoder_q':
            resnet_key = '.'.join(key.split('.')[1:])
            resnet_weight[resnet_key] = moco_weight[key]
    return resnet_weight

class TinyImageNet(Dataset):
    def __init__(self, zip, num_class, transform=None):
        self.zip = zip
        self.root = zip.split('.')[0]
        self.num_class = num_class
        self.transform = transform
        
        if not os.path.exists(self.root):
            # file_name = self.root.split('/')[-1]
            file_path = '/'.join(self.root.split('/')[:-1])
            # print(file_path)
            unzip_file(zip, file_path)
        
        self._make_image_list()
        
        # for i, [img_path, _] in enumerate(self.image):
        #     img = Image.open(img_path).convert('RGB')
        #     self.image[i][0] = self.transform(img)
            
    def _make_image_list(self):
        self.image = []
        root_dir = join(self.root, 'train')
        dir_list = os.listdir(root_dir)
        class_idx = 0
        
        for class_name in dir_list:
            class_path = join(root_dir, class_name)
            if os.path.isdir(class_path):
                temp = join(class_path, 'images')
                images_path = os.listdir(temp)
                self.image.extend([[join(temp, ip), class_idx] for ip in images_path])
                class_idx += 1
                
    def __getitem__(self, index):
        img, lb = self.image[index]
        
        img_q = Image.open(img).convert('RGB')
        img_k = Image.open(img).convert('RGB')
        
        img_q = self.transform(img_q)
        img_k = self.transform(img_k)
        
        return (img_q, img_k), lb

    def __len__(self):
        return len(self.image) 

# class CIFAR(Dataset):
#     def __init__(self, zip, load_type='train', transform=None):
#         self.zip = zip
#         self.load_type = load_type
#         self.root = zip.split('.')[0]
#         self.transform = transform
        
#         if not os.path.exists(self.root):
#             file_path = '/'.join(self.root.split('/')[:-1])
#             unzip_file(zip, file_path)
        
#         self._make_image_list()
            
#     def _make_image_list(self):
#         root_dir = join(self.root, self.load_type)
#         self.img_dict = unpickle(root_dir)
        
#     def __getitem__(self, index):
        
#         total_img = self.img_dict['data']
#         labels    = self.img_dict['fine_labels']
        
#         img, lb = total_img[index], labels[index]
        
#         img = img.reshape(3, 32, 32)
            
#         return img, lb

#     def __len__(self):
#         return len(self.img_dict['data'])

class CIFAR(Dataset):
    def __init__(self, zip, load_type='train', transform=None, val=0.1):
        self.zip = zip
        self.load_type = load_type
        self.root = zip.split('.')[0]
        self.transform = transform
        self.val = val
        
        if not os.path.exists(self.root):
            file_path = '/'.join(self.root.split('/')[:-1])
            unzip_file(zip, file_path)
        
        self._make_image_list()
            
    def _make_image_list(self):
        if self.load_type in ['train', "valid"]:
            root_dir = join(self.root, 'train')
        elif self.load_type == 'test':
            root_dir = join(self.root, 'test')
            
        img_dict = unpickle(root_dir)
        length = len(img_dict['data'])
        
        if self.load_type == 'valid':
            self.total_img = img_dict['data'][:int(length*self.val)]
            self.labels    = img_dict['fine_labels'][:int(length*self.val)]
        elif self.load_type == 'train':
            self.total_img = img_dict['data'][int(length*self.val):]
            self.labels    = img_dict['fine_labels'][int(length*self.val):]
        elif self.load_type == 'test':
            self.total_img = img_dict['data']
            self.labels    = img_dict['fine_labels']
        
    def __getitem__(self, index):
        
        total_img = self.total_img
        labels    = self.labels
        
        img, lb = total_img[index], labels[index]
        
        img = img.reshape(3, 32, 32)
        img = Image.fromarray(np.transpose(img, (1, 2, 0)).astype(np.uint8))
        img = self.transform(img)
        return img, lb

    def __len__(self):
        return len(self.labels)