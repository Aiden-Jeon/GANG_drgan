import os
import torch.nn as nn
from collections import namedtuple

from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as T
from torchvision.utils import save_image

import matplotlib.image as mpimg

class FeatureExtractor(nn.Module):
    def __init__(self, cnn):
        super(FeatureExtractor, self).__init__()
        self.slice1 = nn.Sequential(*list(cnn.features.children())[:4])
        self.slice2 = nn.Sequential(*list(cnn.features.children())[4:32])

    def forward(self, x):
        h = self.slice1(x)
        h_relu_1_2 = h
        h = self.slice2(h)
        h_relu_5_2 = h
        vgg_outputs = namedtuple("Feature", ['relu_1_2', 'relu_5_2'])
        out = vgg_outputs(h_relu_1_2, h_relu_5_2)
        return out
    

class DistortionDataset(Dataset):
    def __init__(self, dis_dir, raw_dir, train=True):
        self.dis_dir = dis_dir
        self.raw_dir = raw_dir
        if train:
            self.dis_files = sorted(os.listdir(dis_dir))[:-3]
            self.raw_files = sorted(os.listdir(raw_dir))[:-3]
        else:
            self.dis_files = sorted(os.listdir(dis_dir))[-3:]
            self.raw_files = sorted(os.listdir(raw_dir))[-3:]
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
    def __getitem__(self, index):
        x = mpimg.imread(self.dis_dir + self.dis_files[index])
        y = mpimg.imread(self.raw_dir + self.raw_files[index])
        x = self.transform(x)
        y = self.transform(y)
        return x, y
    
    def __len__(self):
        return len(self.raw_files)
    

def save_img(realX, targetY, checkpoint_path, epoch):
    G.eval()
    fakeX = G(realX)
    result = torch.cat([realX, fakeX, targetY], 3).cpu().permute(0,2,3,1).data
    result = (result+1)/2
    save_image(result.permute(0,3,1,2), 
              '{0}/img{1}.jpg'.format(checkpoint_path, epoch), nrow=1)