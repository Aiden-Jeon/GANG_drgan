# layers.py
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride, padding, normalize_fn, dropout_fn, activation_fn):
        super(ConvLayer, self).__init__()
        self.features = [nn.Conv2d(input_nc, output_nc, kernel_size, stride, padding)]
        if normalize_fn:
            self.features += [normalize_fn]
        if dropout_fn:
            self.features += [dropout_fn]
        if activation_fn:
            self.features += [activation_fn]
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)
    

class DeconvLayer(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size, stride, padding, normalize_fn, dropout_fn, activation_fn):
        super(DeconvLayer, self).__init__()
        self.features = [nn.ConvTranspose2d(input_nc, output_nc, kernel_size, stride, padding)]
        if normalize_fn:
            self.features += [normalize_fn]
        if dropout_fn:
            self.features += [dropout_fn]
        if activation_fn:
            self.features += [activation_fn]
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        return self.features(x)