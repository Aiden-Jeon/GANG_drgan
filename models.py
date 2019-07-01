# models.py
from layers import *

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, d):
        super(Generator, self).__init__()
        # Encoder
        self.encoder1 = ConvLayer(input_nc=3, output_nc=d, kernel_size=4, stride=2, padding=1, 
            normalize_fn=nn.BatchNorm2d(d), dropout_fn=None, activation_fn=nn.LeakyReLU(0.2, True))         # 256*256 -> 128*128
        self.encoder2 = ConvLayer(d, d*2, 4, 2, 1, nn.BatchNorm2d(d*2), None, nn.LeakyReLU(0.2, True))      # 128*128 -> 64*64
        self.encoder3 = ConvLayer(d*2, d*4, 4, 2, 1, nn.BatchNorm2d(d*4), None, nn.LeakyReLU(0.2, True))    # 64*64 -> 32*32
        self.encoder4 = ConvLayer(d*4, d*8, 4, 2, 1, nn.BatchNorm2d(d*8), None, nn.LeakyReLU(0.2, True))    # 32*32 -> 16*16
        self.encoder5 = ConvLayer(d*8, d*8, 4, 2, 1, nn.BatchNorm2d(d*8), None, nn.LeakyReLU(0.2, True))    # 16*16 -> 8*8
        self.encoder6 = ConvLayer(d*8, d*8, 4, 2, 1, nn.BatchNorm2d(d*8), None, nn.LeakyReLU(0.2, True))    # 8*8 -> 4*4
        self.encoder7 = ConvLayer(d*8, d*8, 4, 2, 1, nn.BatchNorm2d(d*8), None, nn.LeakyReLU(0.2, True))    # 4*4 -> 2*2
        self.encoder8 = ConvLayer(d*8, d*8, 4, 2, 1, nn.BatchNorm2d(d*8), None, nn.ReLU(True))              # 2*2 -> 1*1
        
        # Decoder
        self.decoder1 = DeconvLayer(input_nc=d*8, output_nc=d*8, kernel_size=4, stride=2, padding=1, 
            normalize_fn=nn.BatchNorm2d(d*8), dropout_fn=None, activation_fn=nn.LeakyReLU(0.2, True))           # 1*1 -> 2*2
        self.decoder2 = DeconvLayer(d*8*2, d*8, 4, 2, 1, nn.BatchNorm2d(d*8), None, nn.LeakyReLU(0.2, True))    # 2*2 -> 4*4
        self.decoder3 = DeconvLayer(d*8*2, d*8, 4, 2, 1, nn.BatchNorm2d(d*8), None, nn.LeakyReLU(0.2, True))    # 4*4 -> 8*8
        self.decoder4 = DeconvLayer(d*8*2, d*8, 4, 2, 1, nn.BatchNorm2d(d*8), nn.Dropout2d(0.5), nn.LeakyReLU(0.2, True))   # 8*8 -> 16*16
        self.decoder5 = DeconvLayer(d*8*2, d*4, 4, 2, 1, nn.BatchNorm2d(d*4), nn.Dropout2d(0.5), nn.LeakyReLU(0.2, True))   # 16*16 -> 32*32
        self.decoder6 = DeconvLayer(d*4*2, d*2, 4, 2, 1, nn.BatchNorm2d(d*2), nn.Dropout2d(0.5), nn.LeakyReLU(0.2, True))     # 32*32 -> 64*64
        self.decoder7 = DeconvLayer(d*2*2, d, 4, 2, 1, nn.BatchNorm2d(d), nn.Dropout2d(0.5), nn.LeakyReLU(0.2, True))     # 64*64 -> 128*128
        self.decoder8 = DeconvLayer(d*2, 3, 4, 2, 1, nn.BatchNorm2d(3), None, nn.Tanh())                        # 128*128 -> 256*256

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)        

        d1 = self.decoder1(e8)
        d1 = torch.cat([d1, e7], 1)

        d2 = self.decoder2(d1)
        d2 = torch.cat([d2, e6], 1)

        d3 = self.decoder3(d2)
        d3 = torch.cat([d3, e5], 1)

        d4 = self.decoder4(d3)
        d4 = torch.cat([d4, e4], 1)

        d5 = self.decoder5(d4)
        d5 = torch.cat([d5, e3], 1)

        d6 = self.decoder6(d5)
        d6 = torch.cat([d6, e2], 1)

        d7 = self.decoder7(d6)
        d7 = torch.cat([d7, e1], 1)

        return self.decoder8(d7)

class Discriminator(nn.Module):
    def __init__(self, d):
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            ConvLayer(3, d, 4, 2, 1, None, None, nn.ReLU(True)),                    # 256*256 -> 128*128
            ConvLayer(d, d*2, 4, 2, 1, nn.BatchNorm2d(d*2), None, nn.ReLU(True)),   # 128*128 -> 64*64
            ConvLayer(d*2, d*4, 4, 2, 1, nn.BatchNorm2d(d*4), None, nn.ReLU(True)), # 64*64 -> 32*32
            ConvLayer(d*4, d*8, 4, 2, 1, nn.BatchNorm2d(d*8), None, nn.ReLU(True)), # 32*32 -> 16*16
            ConvLayer(d*8, d*8, 4, 2, 1, nn.BatchNorm2d(d*8), None, nn.ReLU(True)), # 16*16 -> 8*8
            ConvLayer(d*8, d*8, 4, 2, 1, nn.BatchNorm2d(d*8), None, nn.ReLU(True)), # 8*8 -> 4*4
            ConvLayer(d*8, d*8, 4, 2, 1, nn.BatchNorm2d(d*8), None, nn.ReLU(True)), # 4*4 -> 2*2
            ConvLayer(d*8, 1, 4, 2, 1, None, None, nn.Sigmoid())                    # 2*2 -> 1*1
        )

    def forward(self, x):
        return self.features(x)

