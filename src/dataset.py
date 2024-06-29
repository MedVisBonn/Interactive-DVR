import os
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
import nibabel as nib
import nrrd
import random
from omegaconf import OmegaConf

from user_model import UserModel
from utils import *


def get_eval_dataset(
    cfg: OmegaConf,
    initial_annotation: bool = True,
    verbose: bool = False
):  
    if verbose:
        print(f"Loading dataset for subject {cfg.data.subject} with labels {cfg.data.labelset} ...")
    dataset = EvalDataset(
        subject_id=cfg.data.subject, 
        cfg=cfg,
        modality='segmentation',
        to_gpu=False,
        init=cfg["init_mode"]  
    )
    if initial_annotation:
        if verbose:
            print("Sampling initial annotations ...")
        dataset.clear_annotation()
        annot = dataset.initial_annotation(seed=42)
        dataset.update_annotation(annot)
    if verbose:
        print('Done.\n')
    return dataset


def get_train_loader(
    cfg: OmegaConf
):
    data_dir = cfg["data_dir"]
    if cfg.subjects == 'all':
        subject_dirs = [
            d for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d))
            and 'corrupted' not in d
        ]
        n_subjects     = len(subject_dirs)
        train_subjects = subject_dirs[n_subjects//10:]
        val_subjects   = subject_dirs[:n_subjects//10]
        train_loader = DataLoader(
            PretrainingDataset(
                subjects=train_subjects, 
                data_dir=data_dir
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        valid_loader = DataLoader(
            PretrainingDataset(
                subjects=val_subjects, 
                data_dir=data_dir
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    else:
        train_loader = DataLoader(
            PretrainingDataset(
                subjects=cfg.subjects, 
                data_dir=data_dir
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        valid_loader = None

    return train_loader, valid_loader

class PretrainingDataset(Dataset):
    
    def __init__(
        self, 
        subjects: List[str],
        data_dir: str 
    ):
        self.data_dir = data_dir
        self.slice_index = []
        self.mask_index = []
        for subject in subjects:
            subject_dir = f'{data_dir}/{subject}/Diffusion/data'
            slice_files = [
                f'{subject}/Diffusion/data/{f}' for f in os.listdir(subject_dir) 
                if os.path.isfile(os.path.join(subject_dir, f))
            ]
            self.slice_index += slice_files

            mask_files = [
                f'{subject}/Diffusion/brain_mask/{f}' for f in os.listdir(subject_dir)
                if os.path.isfile(os.path.join(subject_dir, f))
            ]
            self.mask_index += mask_files

    def __len__(self):
        return len(self.slice_index)
    
    def __getitem__(self, idx):
        slice_path = f'{self.data_dir}/{self.slice_index[idx]}'
        slice = torch.tensor(nib.load(slice_path).get_fdata()).float()

        mask_path = f'{self.data_dir}/{self.mask_index[idx]}'
        mask = torch.tensor(nib.load(mask_path).get_fdata()).long()


        return {
            'input': slice,
            'mask': mask
        }


class EvalDataset(Dataset):
    
    def __init__(
        self,
        subject_id: str,
        cfg, 
        modality='reconstruction', 
        mode='train', 
        to_gpu=True, 
        init='three_slices', 
    ):
        super().__init__()
        self.cfg          = OmegaConf.to_container(cfg)
        self.axis         = cfg.data.axis
        self.labels       = cfg.data.labels[cfg.data.labelset]
        self.device       = cfg.rank
        self.modality     = modality
        self.mode         = mode
        self.to_gpu       = to_gpu
        self.init         = init

        self.data_path = os.path.join(
            cfg.data.data_dir, 
            str(subject_id), 
            "Diffusion", 
            "data.nii.gz"
        )
        self.data_in = torch.tensor(nib.load(self.data_path).get_fdata()).float()

        self.mask_path = os.path.join(
            cfg.data.data_dir, 
            str(subject_id), 
            "Diffusion", 
            "nodif_brain_mask.nii.gz"
        )
        self.brain_mask = torch.tensor(
            nib.load(self.mask_path).get_fdata(), dtype=torch.bool)
        
        self.tract_path = os.path.join(
            cfg.data.data_dir, 
            str(subject_id), 
            "tracts_masks"
        )
        self.get_tract_masks(self.labels)

        if self.axis == 'coronal':
            self.data_in = self.data_in.permute(1,3,2,0).rot90(1, dims=[2,3])[14:159]
            self.brain_mask = self.brain_mask.permute(1,2,0).rot90(1, dims=[1,2])[14:159]
            self.label = self.label.permute(0,2,3,1).rot90(1, dims=[2,3])[:,14:159]
        else:
            raise NotImplementedError("Only coronal axis is supported at the moment")

        # if cfg['log']:
        #    wandb.config.update({'labels': cfg['labels']})
            
        self.user = UserModel(
            ground_truth=self.label, 
            guidance=cfg["guidance"],
            soft_scores=cfg["soft_scores"],
            cfg=cfg)
            
        # [classes, B, H, W]
        self.annotations = None

        # [B, 1, H, W]
        self.weight = None

        self.pos_weight = (
            len(self.labels)*self.brain_mask.sum() - \
            self.label.sum((1,2,3))[None, :, None, None]
        ) / self.label.sum((1,2,3))[None, :, None, None]
        
        if self.to_gpu:
            self.data_in    = self.data_in.to(self.device)
            self.label      = self.label.to(self.device)
            self.brain_mask = self.brain_mask.to(self.device) 
            self.pos_weight = self.pos_weight.to(self.device) 


    def get_tract_masks(
        self,
        labels: List[str]
    ) -> Tensor:
        self.label = None
        for tract in self.labels:
            
            if f'{tract}.nii.gz' in os.listdir(self.tract_path):
                tract_mask = torch.tensor(
                    nib.load(os.path.join(self.tract_path, f'{tract}.nii.gz')).get_fdata()
                ).long()
                if self.label is None:
                    self.label = tract_mask.unsqueeze(0)
                else:
                    self.label = torch.cat([
                        self.label, 
                        tract_mask.unsqueeze(0)
                    ], dim=0)

            # class other is handled below
            elif tract == 'Other':
                continue
            else:
                # left and right are different classes but the raw data makes
                # even more distictions we don't care for
                if 'left' in tract or 'right' in tract:
                    tract = tract.split('.')[0]
                    tract_parts = tract.split('_')
                    tract, side = tract_parts[0], tract_parts[-1]
                    # tract, side = tract.split('_')
                else:
                    side = ''
                tract_files = [
                    f for f in os.listdir(self.tract_path)
                    if tract in f and side in f
                ]

                tract_mask = None
                for f in tract_files:
                    tract_mask_tmp = torch.tensor(
                        nib.load(os.path.join(self.tract_path, f)).get_fdata()
                    ).long()
                    if tract_mask is None:
                        tract_mask = tract_mask_tmp
                    else:
                        tract_mask = torch.bitwise_or(
                            tract_mask, tract_mask_tmp
                        )

                if self.label is None:
                    self.label = tract_mask.unsqueeze(0)
                else:
                    self.label = torch.cat([
                        self.label, 
                        tract_mask.unsqueeze(0)
                    ], dim=0)


        other = self.brain_mask * ~torch.any(self.label, dim=0)
        self.label = torch.cat([
            other.unsqueeze(0),
            self.label, 
        ], dim=0)
           
        
    def set_mode(self, mode) -> None:
        self.mode = mode


    def set_modality(self, modality) -> None:
        self.modality = modality     
        
        
    def initial_annotation(
        self, 
        seed=42
    ) -> Tensor:
        return self.user.initial_annotation(
            #self.label.detach().cpu(),
            self.cfg["init_voxels"],
            init=self.init, 
            seed=seed
        )
    

    def random_refinement_annotation(
        self,
        prediction, 
        seed=42
    ) -> Tensor:
        
        if self.init == 'per_class':
            mode = 'per_class'
            
        elif self.init == 'three_slices':
            mode = 'single_slice'

        return self.user.random_refinement_annotation(
            prediction, 
            self.annotations.detach().cpu(),
            self.brain_mask.detach().cpu(),
            self.cfg["refinement_voxels"],
            mode=mode,
            seed=seed
        )


    def refinement_annotation(
        self, 
        prediction, 
        uncertainty_map=None, 
        random=False, 
        seed=42
    ) -> Tensor:
        
        if self.init == 'per_class':
            mode = 'per_class'
            
        elif self.init == 'three_slices':
            mode = 'single_slice'
        
        if random:
            return self.user.random_refinement_annotation(
                prediction, 
                self.annotations.detach().cpu(),
                self.brain_mask.detach().cpu(),
                self.cfg["refinement_voxels"],
                mode=mode,
                seed=seed
            )

        else:
            return self.user.refinement_annotation(
                prediction,
                #self.label.detach().cpu(),
                self.annotations.detach().cpu(),
                uncertainty_map,
                self.cfg["refinement_voxels"],
                mode=mode,
                seed=seed
            )


    def update_annotation(self, annotations) -> None:
        assert(annotations.data.type() == 'torch.FloatTensor')

        if self.to_gpu:
            annotations = annotations.to(self.device)

        if self.annotations is None:
            self.annotations = annotations
        else:
            self.annotations += annotations
            self.annotations  = torch.clamp(self.annotations, 0, 1)
            
        self.weight = (self.annotations.sum(0) > 0).unsqueeze(1).float()
        self.pos_weight  = (self.weight.sum() - self.annotations.sum((1,2,3))[None, :, None, None])
#             self.pos_weight  = (1 - self.annotations.sum((1,2,3))[None, :, None, None])
        self.pos_weight /= self.annotations.sum((1,2,3))[None, :, None, None]

    
    def clear_annotation(self) -> None:
        self.annotations = None
        

    def __len__(self) -> int:
        return self.data_in.shape[0]
        

    def __getitem__(self, index) -> dict:

        input_ = self.data_in[index]

        if self.modality == 'reconstruction':
            target = self.data_in[index].detach().clone()
            weight = 1.
            
        elif self.modality == 'segmentation':
        
            if self.mode == 'train':
                target = self.annotations[:, index].detach()
            elif self.mode == 'validate':
                target = self.label[:, index].detach()
    

            weight = self.weight[index]

        mask = self.brain_mask[index]
        
        return {
            'input':  input_,
            'target': target,
            'weight': weight, # may needs unsqueeze(0) in validate
            'mask':   mask
        }
    

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
    