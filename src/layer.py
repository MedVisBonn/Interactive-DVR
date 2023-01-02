import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import interpolate
from math import ceil

from typing import Dict, Iterable, Callable, Generator, Union



class LocalModule(nn.Module):
    def __init__(self, in_channels, out_channels, size=None):
        super().__init__()
        
        self.size = size
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels))
        
    def forward(self, x: Tensor):
        x_out = self.layer(x)
        if self.size is None:
            return [x_out]
        else:
            x_out_low_res = interpolate(x_out, size=(self.size, self.size),
                                        mode='area')
            return [x_out, x_out_low_res]
        
        
        
class RegionalModule(nn.Module):
    def __init__(self, in_channels, out_channels, size=None):
        super().__init__()
        
        self.size = size
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels))    
    
    def forward(self, x: Tensor):
        x_out  = self.layer(x)
        if self.size is None:
            return [x_out]
        else:
            x_out_high_res = interpolate(x_out, size=(self.size, self.size), 
                                         mode='bilinear', align_corners=False)
            return [x_out, x_out_high_res]  
    
    
    
class EncodingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, size_high=None, size_low=None, link=None):
        super().__init__()
        self.link = link

        if link == 'double':
            self.local    = LocalModule(in_channels, out_channels, size_low)
            self.local_pre_conv = nn.Conv2d(out_channels, out_channels, 1)
            self.regional = RegionalModule(in_channels, out_channels, size_high)
            self.regional_pre_conv = nn.Conv2d(out_channels, out_channels, 1)
            
        if link == 'single':
            self.local    = LocalModule(in_channels, out_channels, size_low)
            self.regional_pre_conv = nn.Conv2d(out_channels, out_channels, 1)
            self.regional = RegionalModule(in_channels, out_channels, size_high)
            
        else:
            self.local    = LocalModule(in_channels, out_channels, size_low)
            self.regional = RegionalModule(in_channels, out_channels, size_high)


    def forward(self, x_local: Tensor, x_regional=None):
        if x_regional == None:
            x_regional = x_local
        local    = self.local(x_local)
        regional = self.regional(x_regional)
        
        if (len(local) + len(regional)) == 4:
            local_cat = local[0] + self.regional_pre_conv(regional[1])
            regional_cat = regional[0] + self.local_pre_conv(local[1])
            return local_cat, regional_cat     
        
        elif (len(local) + len(regional)) == 3:
            local_cat = local[0] + self.regional_pre_conv(regional[1])
            return local_cat, regional[0]
        
        else:
            return *local, *regional
        
        
        
class ZeroLinkEncoder(nn.Module):
    def __init__(self, size_high):
        super().__init__()
        self.size = size_high
        
        self.layer0 = EncodingLayer(288, 88)
        self.layer1 = EncodingLayer( 88, 44)
        self.layer2 = EncodingLayer( 44, 22)
        self.layer3 = EncodingLayer( 22, 22)
        self.layer4 = EncodingLayer( 22, 22)
                
    def forward(self, x: Tensor):
        local, regional = self.layer0(x)
        local, regional = self.layer1(local, regional)
        local, regional = self.layer2(local, regional)
        local, regional = self.layer3(local, regional)
        local, regional = self.layer4(local, regional)
        reginal_high_res = interpolate(regional, size=(self.size, self.size), 
                                       mode='bilinear', align_corners=False)
        
        return torch.cat([local, reginal_high_res], dim=1)
    
    
class SingleLinkEncoder(nn.Module):
    def __init__(self, size_high):
        super().__init__()
        
        self.layer0 = EncodingLayer(288, 88, size_high, link='single')        
        self.layer1 = EncodingLayer(88, 44, size_high, link='single')
        self.layer2 = EncodingLayer(44, 22, size_high, link='single')
        self.layer3 = EncodingLayer(22, 22, size_high, link='single')
        
        self.local4            = LocalModule(22, 22)
        self.regional4         = RegionalModule(22, 22, size_high)
        self.regional4_pre_conv = nn.Conv2d(22, 22, 1)
        self.final_local_layer = LocalModule(22, 44, size_high)

    
    def forward(self, x):
        input_local, input_regional = self.layer0(x)
        input_local, input_regional = self.layer1(input_local, input_regional)
        input_local, input_regional = self.layer2(input_local, input_regional)
        input_local, input_regional = self.layer3(input_local, input_regional)
        
        local                = self.local4(input_local)
        _, regional_high_res = self.regional4(input_regional)
        input_final_layer    = local[0] + self.regional4_pre_conv(regional_high_res)
        feature_maps, _      = self.final_local_layer(input_final_layer)

        return feature_maps
        
        
        
class DualLinkEncoder(nn.Module):
    def __init__(self, size_high):
        super().__init__()
        size_low = [ceil(size_high/(2**i)) for i in range(1, 5)]
        
        self.layer0    = EncodingLayer(288, 88, size_high, size_low[0], link='double')
        self.layer1    = EncodingLayer(88, 44, size_high, size_low[1], link='double')
        self.layer2    = EncodingLayer(44, 22, size_high, size_low[2], link='double')
        self.layer3    = EncodingLayer(22, 22, size_high, size_low[3], link='double')
        self.local4    = LocalModule(22, 22, size_high)
        self.regional4 = RegionalModule(22, 22, size_high)
        #self.regional4_pre_conv = nn.Conv2d(22, 22, 1)
        #self.final_local_layer = LocalModule(44, 44, size_high)
        
    def forward(self, x: Tensor) -> Tensor:
        input_local, input_regional = self.layer0(x)
        input_local, input_regional = self.layer1(input_local, input_regional)
        input_local, input_regional = self.layer2(input_local, input_regional)
        input_local, input_regional = self.layer3(input_local, input_regional)

        local, _             = self.local4(input_local)
        _, regional_high_res = self.regional4(input_regional)
        #feature_maps         = torch.cat([local, self.regional4_pre_conv(regional_high_res)], dim=1)
        feature_maps         = torch.cat([local, regional_high_res], dim=1)
        return feature_maps
    
    
    
class LinearTransform(nn.Module):
    """
    A * X + B. A is fixed and global to scale values to a suitable range
    after normalization to mean=0 and var=1. B is a tensor with thresholds,
    one for each output channel, which are added to the normalized inputs to
    shift them before applying soft thresholding via sigmoid.
    """
    def __init__(self, in_channel: int, out_channel: int, translation: Iterable[float], scaling=1.):
        super().__init__()
        translation = torch.tensor(translation, requires_grad=False).view(1, 1, -1, 1, 1)
        scaling     = torch.tensor(scaling, requires_grad=False)
        self.register_buffer('translation', translation)
        self.register_buffer('scaling', scaling)
        
        assert out_channel / in_channel == self.translation.size(2), "number of translations doesn't match number of channel"
        
    def forward(self, x: Tensor) -> Tensor:
        # Input has shape  B, C,     W, H
        # Output has shape B, C * T, W, H
        B, _, W, H    = x.size()
        x_transformed = (x.unsqueeze(2) + self.translation) * self.scaling
        
        return x_transformed.view(B, -1, W, H)
    
    
    
class RandomWarp(nn.Module):
    """
    Combines random Translation and Scaling
    """
    
    def __init__(self, in_channel: int, out_channel: int, steepness=10):
        super().__init__()
        
        warps_per_channel = int(out_channel / in_channel)
        
        translations = torch.rand((1, in_channel, warps_per_channel, 1, 1), \
                                  requires_grad=False) * 2 - 1
        scalings = torch.ones((1, in_channel, warps_per_channel, 1, 1), requires_grad=False)
        scalings *= steepness    
            
        self.register_buffer('translations', translations)
        self.register_buffer('scalings', scalings)
        
    def forward(self, x: Tensor) -> Tensor:
        # Input has shape  B, C,     W, H
        # Output has shape B, C * T, W, H
        B, _, W, H    = x.size()
        x_transformed = (x.unsqueeze(2) + self.translations) * self.scalings
        
        return x_transformed.view(B, -1, W, H)
    

    
class MaskedBatchNorm2d(nn.BatchNorm2d):
    '''
    https://github.com/ptrblck/pytorch_misc/blob/master/batch_norm_manual.py
    '''
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        
    def forward(self, input, mask):
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        
        with torch.no_grad():
            n_voxels = mask.sum()
        # calculate running estimates
        if self.training and n_voxels > 0:
            with torch.no_grad():
                factor = exponential_average_factor# * n_voxels / (9305*8.)
                # move channels to first dim and view with num_features from BatchNorm2d to get mean and var
                masked_input = torch.masked_select(input.transpose(0,1), \
                                                   mask.unsqueeze(0)).view(self.num_features, -1)
                mean         = masked_input.mean(1)
                var          = masked_input.var(1, unbiased=False)
                n            = masked_input.size(1) #masked_input.numel() / masked_input.size(0)

            
                self.running_mean = factor * mean\
                    + (1 - factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = factor * var * n / (n - 1)\
                    + (1 - factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] +\
                                                                  self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


    
class ReconstructionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(44, 88, 5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(88),
            nn.PReLU(88),
            nn.Conv2d(88, 288, 5, stride=1, padding=2, bias=True))
        
    def forward(self, x):
        return self.decoder(x)
    

    
class SegmentationDecoder(nn.Module):
    def __init__(self, n_classes, thresholds='learned', norm='default'):
        super().__init__()
        
        if norm == 'default':
            self.norm = nn.BatchNorm2d(44, affine=False)
        elif norm == 'masked':
            # TODO
            # Current implementation needs a mask and is not giving any benefits
            # self.norm = MaskedBatchNorm2d(44, affine=False)
            raise NotImplementedError
        
        if thresholds == 'learned':
            self.threshold = nn.Sequential(
                nn.Conv2d(44, 44*n_classes, 1, groups=44, bias=True),
                nn.Sigmoid())
            
        elif thresholds == 'selected':
            self.threshold = nn.Sequential(
                LinearTransform(44, 44*n_classes, 
                                [-0.9674, -0.4307, 0.0, 0.4307, 0.9674], 10.), 
                nn.Sigmoid())
            
        elif thresholds == 'random':
            self.threshold = nn.Sequential(
                RandomWarp(44, 44*n_classes), 
                nn.Sigmoid())
            
        elif thresholds == 'old':
            self.threshold = nn.Sequential(      
                nn.Conv2d(44, 22, 1),
                nn.PReLU(22),
            )
            
        elif thresholds == 'old2':
            self.threshold = nn.Sequential(
                nn.Conv2d(44, 22, 5, stride=1, padding=2),
                nn.PReLU(22),
                nn.BatchNorm2d(22),

                nn.Conv2d(22, 22, 1),
                nn.PReLU(22),
                nn.BatchNorm2d(22),
            )
        
        
        if thresholds == 'old':
            self.aggregate =  nn.Conv2d(22, n_classes, 1)
            
        elif thresholds == 'old2':
            self.aggregate =  nn.Conv2d(22, n_classes, 5, stride=1, padding=2) 
            
        else:
            self.aggregate = nn.Conv2d(44*n_classes, n_classes, 
                                       1, stride=1, padding=0, bias=False)

        
    def forward(self, x):
        x_norm      = self.norm(x)
        x_thresholded = self.threshold(x_norm)
        x_aggregated  = self.aggregate(x_thresholded)
        return x_aggregated# , x_thresholded
    
