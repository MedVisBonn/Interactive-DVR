import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
import nibabel as nib
import nrrd
import random

from user_model import UserModel
from utils import *

class AEDataset(Dataset):
    
    def __init__(self, cfg, modality='reconstruction', mode='train', set=1,
                 augment=False, to_gpu=True, paper_init=False, smooth_label=False):
        
        self.cfg         = cfg
        self.cfg['rank'] = 0
        self.modality    = modality
        self.mode        = mode
        self.augment     = augment
        self.to_gpu      = to_gpu
        self.paper_init  = paper_init
        self.smooth_label = smooth_label
    
        
        try:
            data_in = torch.load(cfg["data_dir"] + "data_in.pt")
        except:
            data_in = torch.tensor(nib.load(cfg["data_path"]).get_fdata()).float()
            torch.save(data_in, cfg["data_dir"] + "data_in.pt")        
        
        # shape [145, 288, 145, 145]  [B, C, H, W]
        self.input = (data_in / data_in.amax(dim=(0,1,2))).permute(1,3,0,2)[14:159]

        # shape [145, 145, 145]       [B, H, W]
        self.brain_mask = torch.tensor(
            nib.load(cfg["active_mask_path"]).get_fdata(), dtype=torch.bool).permute(1,0,2)[14:159]

        # shape [12, 145, 145, 145]    [classes, B, H, W]
        self.tract_masks = torch.load(cfg['data_dir'] + 'tract_masks/complete.pt')
        if set == 1:
            cfg['labels'] = ["Other", "CST"]
            self.label = torch.cat([self.brain_mask.unsqueeze(0).float() - \
                                    self.tract_masks[2:3], self.tract_masks[2:3]], \
                                   dim=0).bool()
        elif set == 2:
            cfg['labels'] = ["Other", "CG", "CST", "FX", "CC"]
            self.label = self.tract_masks[:5]
        elif set == 3:
            cfg['labels'] = ["Other", "IFO_left", "IFO_right", "ILF_left", \
                             "ILF_right", "SLF_left", "SLF_right"]
            self.label = self.tract_masks[5:]
            
        self.user = UserModel(self.label)
            
        # [classes, B, H, W]
        self.annotations = None

        # [B, 1, H, W]
        self.weight = None

        self.pos_weight = (5*self.brain_mask.sum() - \
                           self.label.sum((1,2,3))[None, :, None, None]) / \
                           self.label.sum((1,2,3))[None, :, None, None]
        
        if self.to_gpu:
            self.input      = self.input.to(self.cfg["rank"])
            self.label      = self.label.to(self.cfg["rank"])
            self.brain_mask = self.brain_mask.to(self.cfg["rank"]) 
            self.pos_weight = self.pos_weight.to(self.cfg["rank"]) 
        
        if self.augment:
            ### Init augmentations
            self.inpaint = InPainting(3)
            self.outpaint = OutPainting(5)
            
            ### call augmentation calculation routine here
            self.update_painting()
           
        
    def set_mode(self, mode) -> None:
        self.mode = mode


    def set_modality(self, modality) -> None:
        self.modality = modality     
        
        
    def initial_annotation(self) -> Tensor:
        return self.user.initial_annotation(#self.label.detach().cpu(),
                                             self.cfg["init_voxels"],
                                             self.paper_init)


    def refinement_annotation(self, prediction) -> Tensor:
        return self.user.refinement_annotation(prediction,
                                                #self.label.detach().cpu(),
                                                self.annotations.detach().cpu(),
                                                self.cfg["refinement_voxels"])


    def update_annotation(self, annotations) -> None:
        assert(annotations.data.type() == 'torch.FloatTensor')

        if self.to_gpu:
            annotations = annotations.to(self.cfg["rank"])

        if self.annotations is None:
            self.annotations = annotations
        else:
            self.annotations += annotations
            self.annotations  = torch.clamp(self.annotations, 0, 1)
        
        if self.smooth_label:
            self.smooth_annotations = (self.annotations.clone()*(1. - 0.05)) + 0.01
        
        # repeat(1,len(self.cfg["labels"]),1,1)
        self.weight = (self.annotations.sum(0) > 0).unsqueeze(1).float()
        self.pos_weight  = (self.weight.sum() - self.annotations.sum((1,2,3))[None, :, None, None])
        self.pos_weight /= self.annotations.sum((1,2,3))[None, :, None, None]
        
    
    def clear_annotation(self) -> None:
        self.annotations = None
        
        
    def update_painting(self, k_in=10, k_out=15) -> None:
        assert self.inpaint is not None, "Init inpaint first"
        assert self.outpaint is not None, "Init outpaint first"
        
        self.inpaint.update()
        self.outpaint.update()
        
        # Random choice between in and out painting, ramdomly applied
        random_choice       = transforms.RandomChoice([self.inpaint, self.outpaint])
        random_apply_lambda = lambda slc: random_choice(slc) if torch.rand(1) > 0.1 else slc
        self.transform      = transforms.Lambda(random_apply_lambda)
        
        
    def __getitem__(self, index) -> dict:
        
        input_ = self.input[index]
        if self.augment:
            input_ = self.transform(input_)
        
        if self.modality == 'reconstruction':
            target = self.input[index].detach().clone()
            weight = 1.
            
        elif self.modality == 'segmentation':
            if self.mode == 'train':
                target = self.annotations[:, index]
            elif self.mode == 'validate':
                target = self.label[:, index]    
            weight = self.weight[index]
            
        mask = self.brain_mask[index]
        
        return {'input':  input_,
                'target': target,
                'weight': weight, # may needs unsqueeze(0) in validate
                'mask':   mask} 


    def __len__(self) -> int:
        return self.input.shape[0]  