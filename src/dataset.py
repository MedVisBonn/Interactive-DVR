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
    
    def __init__(self, cfg, modality='reconstruction', mode='train', set=1, normalize=True,
                 augment=False, localize=False, balance=False, to_gpu=True, init='three_slices', smooth_label=False):
        
        self.cfg          = cfg
        self.cfg['rank']  = 0
        self.modality     = modality
        self.mode         = mode
        self.augment      = augment
        self.to_gpu       = to_gpu
        self.init         = init
        self.smooth_label = smooth_label
        self.localize     = localize
        self.balance      = balance
        
        try:
            data_in = torch.load(cfg["data_dir"] + "data_in.pt")
        except:
            data_in = torch.tensor(nib.load(cfg["data_path"]).get_fdata()).float()
            torch.save(data_in, cfg["data_dir"] + "data_in.pt")        
        
        # shape [145, 288, 145, 145]  [B, C, H, W]
        self.normalize = normalize
        if normalize:
            self.input = (data_in / data_in.amax(dim=(0,1,2))).permute(1,3,0,2)[14:159]
        else:
            self.input = data_in.permute(1,3,0,2)[14:159]
        
        # shape [145, 145, 145]       [B, H, W]
        self.brain_mask = torch.tensor(
            nib.load(cfg["active_mask_path"]).get_fdata(), dtype=torch.bool).permute(1,0,2)[14:159]

        # shape [12, 145, 145, 145]    [classes, B, H, W]
        self.tract_masks = torch.load(cfg['data_dir'] + 'tract_masks/complete.pt')
        
        self.set = set
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

            
        #if cfg['log']:
        #    wandb.config.update({'labels': cfg['labels']})
            
        self.user = UserModel(self.label)
            
        # [classes, B, H, W]
        self.annotations = None

        # [B, 1, H, W]
        self.weight = None

        self.pos_weight = (len(cfg['labels'])*self.brain_mask.sum() - \
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
            
        if self.localize:
            self.grad = torch.linspace(0, 1, 145).repeat(1,145,1)
            if self.to_gpu:
                self.grad = self.grad.to(self.cfg['rank'])
                

    def get_index_tensor_for_batching(self) -> None:
        # get indices for all non-empty slices by class
        idx = torch.nonzero(self.annotations.cpu().sum(axis=(2,3)))
        # sort indices by number of annotations per slice for each class
        indices_sorted_num_annotations_per_class = [
            self.annotations[c, idx[idx[:, 0]==c, 1]].sum((-1, -2)).sort(descending=True)[1]
            for c in range(len(self.cfg['labels']))
        ]
        # get indices per class and sort them according to index above. If we repeat short 
        # index lists later, we make sure to repeat slices with many annotations first
        slice_indices_per_class = [
            idx[idx[:, 0] == c, 1][indices_sorted_num_annotations_per_class[c]]
            for c in range(len(self.cfg['labels']))
        ]
        # count slices per class
        slices_per_class = [len(c) for c in slice_indices_per_class]
        # sort classes by number of annotations. Classes with more annotations loose
        # duplicated indices first.
        classes_descending_by_annots = torch.Tensor(slices_per_class).sort(descending=True)[1]
        # each slice is only used once. Class with least amount of annotations keeps
        # the slice and its removed for every other class
        for i, c in enumerate(classes_descending_by_annots):
            slice_indices_per_class[c] = slice_indices_per_class[c].tolist()
            for idx in slice_indices_per_class[c]:
                if any(idx in slice_indices_per_class[c_] for c_ in classes_descending_by_annots[i+1:]):
                    slice_indices_per_class[c].remove(idx)
        # convert slices indices back to list of tensors
        slice_indices_per_class = [torch.Tensor(c) for c in slice_indices_per_class]
        # update the slices_per_class after removing duplicates
        slices_per_class_short = [len(c) for c in slice_indices_per_class]
        # save max length for cutting
        max_length = max(slices_per_class_short)
        # calculate how often each class index list has to be repeated to be larger
        # than the largest class index list
        repeat_per_class = [-(-max_length//len_) for len_ in slices_per_class_short]
        # stack results. For each class repeat first if its not the longest list.
        # After, cut any index that makes the list longer than the longest list, starting
        # from the end.
        index_tensor = torch.stack(
            [
                slice_idxs.repeat(repeats)[:max_length]
#                 torch.cat(
#                     [
#                         slice_idxs, torch.Tensor(
#                             [slice_idxs.repeat(2)[0]] * max_length
#                         )
#                     ]
#                 )[:max_length]
                for slice_idxs, repeats 
                in zip(slice_indices_per_class, repeat_per_class)
            ]
        )
        # has shape [n_classes, length_of_longest_index_list]
        self.index_tensor = index_tensor.long()
                        

    def permute_index_tensor(self) -> None:
        # check if index tensor exists
        assert self.index_tensor is not None, "Built index tensor before permuting it"
        # get permuted indices for each class
        idxs = torch.argsort(torch.rand(*self.index_tensor.shape), dim=-1)
        # sort each class independently
        self.index_tensor = torch.gather(self.index_tensor, dim=-1, index=idxs)
           
        
    def set_mode(self, mode) -> None:
        self.mode = mode


    def set_modality(self, modality) -> None:
        self.modality = modality     
        
        
    def initial_annotation(self, seed=42) -> Tensor:
        return self.user.initial_annotation(#self.label.detach().cpu(),
                                             self.cfg["init_voxels"],
                                             init=self.init, 
                                             seed=seed)


    def refinement_annotation(self, prediction, seed=42) -> Tensor:
        
        if self.init == 'per_class':
            mode = 'per_class'
            
        if self.init == 'three_slices':
            mode = 'single_slice'
        
        
        return self.user.refinement_annotation(prediction,
                                               #self.label.detach().cpu(),
                                               self.annotations.detach().cpu(),
                                               self.cfg["refinement_voxels"],
                                               mode=mode,
                                               seed=seed)


    def update_annotation(self, annotations) -> None:
        assert(annotations.data.type() == 'torch.FloatTensor')

        if self.to_gpu:
            annotations = annotations.to(self.cfg["rank"])

        if self.annotations is None:
            self.annotations = annotations
        else:
            self.annotations += annotations
            self.annotations  = torch.clamp(self.annotations, 0, 1)
            
        if self.balance:
            self.get_index_tensor_for_batching()
            self.permute_index_tensor()
        
        if self.smooth_label:
            self.smooth_annotations = (self.annotations.clone()*(1. - 0.05)) + 0.01
        
        # repeat(1,len(self.cfg["labels"]),1,1)
        
        if self.balance:
            self.weight = (self.annotations.sum(0) > 0).unsqueeze(1).float()
            self.cls_weight = (self.weight[self.index_tensor.flatten()].sum() - 
                                 self.annotations[:, self.index_tensor.flatten()].sum((1,2,3))[None, :, None, None])
#             self.pos_weight  = (1 -
#                                 self.annotations[:, self.index_tensor.flatten()].sum((1,2,3))[None, :, None, None])
            self.pos_weight = self.cls_weight / self.annotations[:, self.index_tensor.flatten()].sum((1,2,3))[None, :, None, None]

        else:
            self.weight = (self.annotations.sum(0) > 0).unsqueeze(1).float()
            self.pos_weight  = (self.weight.sum() - self.annotations.sum((1,2,3))[None, :, None, None])
#             self.pos_weight  = (1 - self.annotations.sum((1,2,3))[None, :, None, None])
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
        if self.balance:
            input_ = self.input[self.index_tensor[:, index]]
        else:
            input_ = self.input[index]
            
        if self.augment:
            input_ = self.transform(input_)
        
        if self.modality == 'reconstruction':
            target = self.input[index].detach().clone()
            weight = 1.

            if self.localize:
                target = torch.cat([target, self.grad, self.grad.transpose(-1, -2)])
            
        elif self.modality == 'segmentation':
        
            if self.mode == 'train':
                if self.balance:
                    target = self.annotations[:, self.index_tensor[:, index]].detach()
                else:
                    target = self.annotations[:, index].detach()
            elif self.mode == 'validate':
                target = self.label[:, index].detach()
                
            if self.balance:
                weight = self.weight[self.index_tensor[:, index]]
            else:
                weight = self.weight[index]
        
        if self.balance:
            mask = self.brain_mask[self.index_tensor[:, index]]
        else:
            mask = self.brain_mask[index]
        
        return {'input':  input_,
                'target': target,
                'weight': weight, # may needs unsqueeze(0) in validate
                'mask':   mask} 
    

    def __len__(self) -> int:
        if self.balance:
            return self.index_tensor.shape[1]
        else:
            return self.input.shape[0]
    
    
class AEValidationSet(Dataset):
    pass
    