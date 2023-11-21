#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:02:19 2020

@author: Jonathan Lennartz
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, Tensor
import torch.nn.functional as F
import nibabel as nib
from os.path import join
from pathlib import Path
from time import time
from typing import List, Dict, Iterable, Callable, Generator, Union
#import wandb
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import precision_recall_fscore_support
from sklearn.decomposition import PCA
from scipy.ndimage import distance_transform_edt
import random
import copy
from datetime import date
from os import makedirs, path
from copy import deepcopy
from tqdm.auto import tqdm

def debugging(message):
    print("".center(60, "#"))
    print(message.center(60, "-"))
    print("".center(60, "#"))
    print("\n")
    print("\n")


###############################################################################
################################# TRAIN MODEL #################################
###############################################################################

def count_model_parameters(model: nn.Module) -> Dict[str, int]:
    """Counts total parameters of model.

    Args:
      model: Pytorch model

    Returns:
        The number of total and trainable parameters.
    """

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {'total': total, 'trainable': trainable}


def model_param_sum(model: nn.Module) -> int:
    s = 0
    with torch.no_grad():
        for p in model.parameters():
            s += p.sum().item()
    return s


def weight_reset(model: nn.Module) -> None:
    if isinstance(model, torch.nn.Conv2d) or isinstance(model, torch.nn.Linear):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        model.reset_parameters()
        
        
def save_model(model: nn.Module, p: str) -> None:
    torch.save(model.state_dict(), p)

    
def load_model(model: nn.Module, p: str, same=True) -> nn.Module:
    state_dict = torch.load(p)
    if same:
        model.load_state_dict(state_dict, strict=True)

    else:
        state_dict_tmp = state_dict.copy()
        for key in state_dict_tmp.keys():
            if 'decoder' in key:
                del state_dict[key]
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print(missing_keys)
        
    return model


def make_path(obj: object, it: str, kind: str, modality='reconstruction') -> str:
    
    p = '/'.join(['../experiments', str(date.today()), modality, kind]) + '/'
 
    if not path.exists(p):
        makedirs(p)
    
    return p + obj.__class__.__name__ + '-' + str(it) + '.pt'


def epoch_average(losses, counts):
    losses_np = np.array(losses)
    counts_np = np.array(counts)
    weighted_losses = losses_np * counts_np
    return weighted_losses.sum()/counts_np.sum()


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=7):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True
        return False


def balanced_collate(batch):
    batch = default_collate(batch)
    for key in batch:
        if key == 'target':
            batch[key] = batch[key].permute(0,2,1,3,4).flatten(0,1)
        else:
            batch[key] = batch[key].flatten(0,1)
    return batch
    
    
###############################################################################
############################### Transformations ###############################
###############################################################################    



class PaintBlock(object):
    
    def __init__(self):
        self.anchors      = None
        self.block_sizes  = None
    
    @torch.no_grad()
    def set_anchors(self, anchors: Tensor) -> None:
        self.anchors = anchors
    
    @torch.no_grad()
    def set_block_sizes(self, block_sizes: Tensor) -> None:
        self.block_sizes = block_sizes    
    
    
class InPainting(nn.Module):
    
    def __init__(self, k=3, slc_size=145, gamma=0.1):
        super().__init__()
        self.k = k
        self.slc_size = slc_size
        self.gamma = gamma
        
        self.update()
    
    
    @torch.no_grad()
    def update(self) -> None:
        self.blocks = []
        for _ in range(self.k):
            block = PaintBlock()
            block_size_x, block_size_y = torch.randint(low  = self.slc_size//6,
                                                       high = self.slc_size//3,
                                                       size = (2,))
            
            anchor_x = torch.randint(low  = 3, high = self.slc_size-block_size_x-3, size=(1,))
            anchor_y = torch.randint(low  = 3, high = self.slc_size-block_size_y-3, size=(1,))
            
            block.set_block_sizes([block_size_x, block_size_y])
            block.set_anchors([anchor_x, anchor_y])
            
            self.blocks.append(block)
            
            
    @torch.no_grad()        
    def forward(self, slc_orig: Tensor) -> Tensor:
        slc = slc_orig.detach().clone()
        
        with torch.no_grad():
            for i in range(self.k):
                anchor_x, anchor_y         = self.blocks[i].anchors
                block_size_x, block_size_y = self.blocks[i].block_sizes
                
                slc[:, anchor_x:anchor_x+block_size_x,
                       anchor_y:anchor_y+block_size_y] = torch.rand(size=(block_size_x, 
                                                                          block_size_y), device = slc.device) * self.gamma
            return slc
            

class OutPainting(nn.Module):
    
    def __init__(self, k=5, slc_size=145, gamma=0.1):
        super().__init__()
        self.k = k
        self.slc_size = slc_size
        self.gamma = gamma
        
        self.update()
     
    
    @torch.no_grad()
    def update(self) -> None:
        self.blocks = []
        
        for _ in range(self.k):
            block = PaintBlock()
            block_size_x, block_size_y = self.slc_size - torch.randint(low  = 3*self.slc_size//7,
                                                                       high = 4*self.slc_size//7,
                                                                       size = (2,))

            anchor_x = torch.randint(low  = 3, high = self.slc_size-block_size_x-3, size=(1,))
            anchor_y = torch.randint(low  = 3, high = self.slc_size-block_size_y-3, size=(1,))
            
            block.set_block_sizes([block_size_x, block_size_y])
            block.set_anchors([anchor_x, anchor_y])
            
            self.blocks.append(block)
            
            
    @torch.no_grad()       
    def forward(self, slc_orig: Tensor) -> Tensor:
        device = slc_orig.device
        slc = torch.ones_like(slc_orig)
        slc *= torch.rand(size=slc.shape[-2:], device=device) * self.gamma
        
        with torch.no_grad():
            for i in range(self.k):
                anchor_x, anchor_y         = self.blocks[i].anchors
                block_size_x, block_size_y = self.blocks[i].block_sizes               
                
                slc[:, anchor_x:anchor_x+block_size_x,
                       anchor_y:anchor_y+block_size_y] = slc_orig[:, anchor_x:anchor_x+block_size_x,
                                                                     anchor_y:anchor_y+block_size_y]
            return slc


###############################################################################
################################## EVALUATION #################################
###############################################################################


class OutputHook:
    def __call__(self, module, input, output):
        self.output = output

@torch.no_grad()
class FeatureExtractor(nn.Module):
    # https://gist.github.com/fkodom/27ed045c9051a39102e8bcf4ce31df76#file-feature_extractor_hook-py
    def __init__(self, model: nn.Module, layers: Iterable[str]):
        super().__init__()
        self.model = deepcopy(model)
        self.layers = layers
        self._features = {layer: None for layer in layers}

        for layer_id in layers:
            layer = dict([*self.model.named_modules()])[layer_id]
            layer.register_forward_hook(self.save_outputs_hook(layer_id))
            

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            if self._features[layer_id] is None:
                self._features[layer_id] = output.cpu()
            else:
                self._features[layer_id] = torch.cat([self._features[layer_id], output.cpu()], dim=0)
                
        return fn        
        

    def forward(self, dataset: Dataset) -> Dict[str, Tensor]:
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
        
        with torch.no_grad():
            for batch in dataloader:
                in_ = batch['input']
                if next(self.model.parameters()).device.type == 'cuda' and in_.device.type == 'cpu':
                    in_ = in_.to(0)
                _ = self.model(in_)
                
        return self._features



def evaluate_RF(dataset: Dataset, features: Tensor, cfg: Dict[str, str]) \
                -> Union[Dict[str, float], Tensor]:

                ###############################
                ##### TRAIN RANDOM FOREST #####
                ###############################
                
    train_mask  = dataset.weight.detach().cpu().squeeze().numpy()    #.permute(0,2,3,1).repeat(1,1,1,44).numpy()
    test_mask   = dataset.brain_mask.detach().cpu().numpy()#.unsqueeze(3).repeat(1,1,1,44).numpy()
    train_label = dataset.annotations.detach().cpu().permute(1,2,3,0).numpy()
    test_label  = dataset.label.detach().cpu().permute(1,2,3,0).numpy()
    
    # Input - Mask voxels that are not labelled before flattening the input
    X_train = features.reshape((-1, features.shape[-1]))[train_mask.reshape(-1) == 1]
    X_test  = features.reshape((-1, features.shape[-1]))[test_mask.reshape(-1)  == 1]
    # Target - Same as above. Mask before flattening
    Y_train = train_label.reshape((-1, train_label.shape[-1]))[train_mask.reshape(-1) == 1]
    Y_test  = test_label.reshape((-1,  train_label.shape[-1]))[test_mask.reshape(-1)  == 1]

    # Init Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100,
                         bootstrap=True,
                         oob_score=True,
                         random_state=0,
                         n_jobs=-1,
                         max_features="sqrt", # changed from "auto" because auto got removed
                         class_weight="balanced",
                         max_depth=None,
                         min_samples_leaf=cfg["min_samples_leaf"])

    # Train
    clf.fit(X_train, Y_train)
    # predict labels in test mask
    predicted_prob    = clf.predict_proba(X_test)
    Y_predicted_prob  = torch.tensor(np.array([p[:, 1] for p in predicted_prob])).T
    Y_predicted_label = (Y_predicted_prob > 0.5)*1


                ###############################
                ####### Save Prediction #######
                ###############################
                
    n_classes = len(cfg['labels'])
    prediction = torch.zeros((145,145,145, n_classes))
    prediction.view(-1, n_classes)[test_mask.reshape(-1)  == 1] = Y_predicted_label.float()

                ###############################
                ##### Evaluate Prediction #####
                ###############################

    # Constant for numerical stability
    eps = 1e-5
    # statictics for precision, recall and Dice (f1)
    TP       = (Y_predicted_label * Y_test).sum(axis=0)
    TPplusFP = Y_predicted_label.sum(axis=0)
    TPplusFN = Y_test.sum(axis=0)

    precision = (TP + eps) / (TPplusFP + eps)
    recall    = (TP + eps) / (TPplusFN + eps)
    f1        = (2 * precision * recall + eps) / ( precision + recall  + eps)

    labels = cfg["labels"]
    scores = {}
    for c in range(len(labels)):
        scores[f"{labels[c]}_precision"] = precision[c].numpy()
        scores[f"{labels[c]}_recall"] = recall[c].numpy()
        scores[f"{labels[c]}_f1"] = f1[c].numpy()

    scores["Avg_prec_tracts"] = precision[1:].mean().numpy()
    scores["Avg_recall_tracts"] = recall[1:].mean().numpy()
    scores["Avg_f1_tracts"] = f1[1:].mean().numpy()
    
    

    return scores, prediction.permute(3,0,1,2)


def calc_binary_entropy(prob):
    eps = 1e-5
    
    #Edge Cases
    prob = torch.clamp(prob, eps, 1-eps)
    
    return -(prob*torch.log2(prob) + (1-prob)*torch.log2(1-prob))


def uncertainty_entropy(Y_predicted_prob, n_classes, test_mask):
    entropy = calc_binary_entropy(Y_predicted_prob)
    entropy_map = torch.zeros((145,145,145, n_classes))
    entropy_map[test_mask  == 1] = entropy.float()
    mean_entropy_map = entropy_map.mean(axis=-1)
    return entropy_map, mean_entropy_map


def uncertainty_sd(train_label, test_mask, n_classes):
    # all classes
    annotated_voxels = torch.any(torch.from_numpy(train_label), dim=-1)
    spatial_distances = torch.tensor(distance_transform_edt(~annotated_voxels))
    spatial_distance_map = torch.zeros((145,145,145))
    spatial_distance_map[test_mask == 1] = spatial_distances[test_mask == 1].float()
    # per class
    sd_map_per_class = torch.zeros((n_classes, 145,145,145))
    for i in range(n_classes):
        train_label_i = ~torch.tensor(train_label[:,:,:,i], dtype=torch.int8)
        sd = torch.tensor(distance_transform_edt(train_label_i))
        sd_map_per_class[i, test_mask == 1] = sd[test_mask == 1].float()
    return spatial_distance_map, sd_map_per_class.permute(1,2,3,0)


def uncertainty_fd(train_label, features, test_mask, n_classes):
    
    def compute_anomaly_scores(annotated_features, mask):
        iforest = IsolationForest(n_estimators=100, random_state=0, n_jobs=-1).fit(annotated_features)
        anomaly_scores = iforest.decision_function(features[mask].reshape(-1, 44))
        return torch.from_numpy(1 - (anomaly_scores + 0.5)).float()
    
    # all classes
    annotated_voxels = torch.any(torch.from_numpy(train_label), dim=-1)
    annotated_features = features[annotated_voxels].reshape(-1, 44)
    brain_na_mask = (torch.from_numpy(test_mask == 1)) & (annotated_voxels == 0)
    anomaly_scores_map = torch.zeros((145,145,145))
    anomaly_scores_map[brain_na_mask] = compute_anomaly_scores(annotated_features, brain_na_mask)

    # per class
    fd_map_per_class = torch.zeros((n_classes, 145,145,145))
    for i in range(n_classes):
        train_label_i = torch.tensor(train_label[:,:,:,i], dtype=torch.int8)
        annotated_features = features[train_label_i.bool()].reshape(-1, 44)
        brain_na_mask = (torch.from_numpy(test_mask == 1)) & (train_label_i == 0)
        fd_map_per_class[i, brain_na_mask == 1] = compute_anomaly_scores(annotated_features, brain_na_mask)

    return anomaly_scores_map, fd_map_per_class.permute(1,2,3,0)




def evaluate_RF_with_uncertainty(dataset: Dataset, features: Tensor, cfg: Dict[str, str]) \
                -> Union[Dict[str, float], Tensor]:

                ###############################
                ##### TRAIN RANDOM FOREST #####
                ###############################
                
    train_mask  = dataset.weight.detach().cpu().squeeze().numpy()    #.permute(0,2,3,1).repeat(1,1,1,44).numpy()
    test_mask   = dataset.brain_mask.detach().cpu().numpy()#.unsqueeze(3).repeat(1,1,1,44).numpy()
    train_label = dataset.annotations.detach().cpu().permute(1,2,3,0).numpy()
    test_label  = dataset.label.detach().cpu().permute(1,2,3,0).numpy()
    
    # Input - Mask voxels that are not labelled before flattening the input
    X_train = features.reshape((-1, features.shape[-1]))[train_mask.reshape(-1) == 1]
    X_test  = features.reshape((-1, features.shape[-1]))[test_mask.reshape(-1)  == 1]
    # Target - Same as above. Mask before flattening
    Y_train = train_label.reshape((-1, train_label.shape[-1]))[train_mask.reshape(-1) == 1]
    Y_test  = test_label.reshape((-1,  train_label.shape[-1]))[test_mask.reshape(-1)  == 1]

    # Init Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100,
                         bootstrap=True,
                         oob_score=True,
                         random_state=0,
                         n_jobs=-1,
                         max_features="sqrt", # changed from "auto" because auto got removed
                         class_weight="balanced",
                         max_depth=None,
                         min_samples_leaf=cfg["min_samples_leaf"])

    # Train
    clf.fit(X_train, Y_train)
    # predict labels in test mask
    predicted_prob    = clf.predict_proba(X_test)
    Y_predicted_prob  = torch.tensor(np.array([p[:, 1] for p in predicted_prob])).T
    Y_predicted_label = (Y_predicted_prob > 0.5)*1


                ###############################
                ####### Save Prediction #######
                ###############################
                
    n_classes = len(cfg['labels'])
    prediction = torch.zeros((145,145,145, n_classes))
    prediction.view(-1, n_classes)[test_mask.reshape(-1)  == 1] = Y_predicted_label.float()

                ###############################
                ##### Evaluate Prediction #####
                ###############################

    # Constant for numerical stability
    eps = 1e-5
    # statictics for precision, recall and Dice (f1)
    TP       = (Y_predicted_label * Y_test).sum(axis=0)
    TPplusFP = Y_predicted_label.sum(axis=0)
    TPplusFN = Y_test.sum(axis=0)

    precision = (TP + eps) / (TPplusFP + eps)
    recall    = (TP + eps) / (TPplusFN + eps)
    f1        = (2 * precision * recall + eps) / ( precision + recall  + eps)


    # uncertainty (entropy)
    entropy_map, mean_entropy_map = uncertainty_entropy(Y_predicted_prob, n_classes, test_mask)

    # uncertainty (spatial distance)
    sd, sd_per_class = uncertainty_sd(train_label, test_mask, n_classes)

    # uncertainty (feature distance)
    fd, fd_per_class = uncertainty_fd(train_label, features, test_mask, n_classes)


    labels = cfg["labels"]
    scores = {}
    for c in range(len(labels)):
        scores[f"{labels[c]}_precision"] = precision[c].numpy()
        scores[f"{labels[c]}_recall"] = recall[c].numpy()
        scores[f"{labels[c]}_f1"] = f1[c].numpy()

    scores["Avg_prec_tracts"] = precision[1:].mean().numpy()
    scores["Avg_recall_tracts"] = recall[1:].mean().numpy()
    scores["Avg_f1_tracts"] = f1[1:].mean().numpy()
    
    
    return scores, prediction.permute(3,0,1,2), entropy_map.permute(3,0,1,2), mean_entropy_map, \
        sd, sd_per_class.permute(3,0,1,2), fd, fd_per_class.permute(3,0,1,2)


def evaluate_RF_tmp(dataset: Dataset, features: Tensor, prev_correct: Tensor,
                    cfg: Dict[str, str]) -> Union[Dict[str, float], Tensor]:

                ###############################
                ##### TRAIN RANDOM FOREST #####
                ###############################
                
    train_mask  = dataset.weight.detach().cpu().squeeze().numpy()    #.permute(0,2,3,1).repeat(1,1,1,44).numpy()
    test_mask   = dataset.brain_mask.detach().cpu().numpy()#.unsqueeze(3).repeat(1,1,1,44).numpy()
    train_label = dataset.annotations.detach().cpu().permute(1,2,3,0).numpy()
    test_label  = dataset.label.detach().cpu().permute(1,2,3,0).numpy()
    
    # Input - Mask voxels that are not labelled before flattening the input
    X_train = features.reshape((-1, features.shape[-1]))[train_mask.reshape(-1) == 1]
    X_test  = features.reshape((-1, features.shape[-1]))[test_mask.reshape(-1)  == 1]
    # Target - Same as above. Mask before flattening
    Y_train = train_label.reshape((-1, train_label.shape[-1]))[train_mask.reshape(-1) == 1]
    Y_test  = test_label.reshape((-1,  train_label.shape[-1]))[test_mask.reshape(-1)  == 1]

    # Init Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100,
                         bootstrap=True,
                         oob_score=True,
                         random_state=0,
                         n_jobs=-1,
                         max_features="auto",
                         class_weight="balanced",
                         max_depth=None,
                         min_samples_leaf=cfg["min_samples_leaf"])

    # Train
    clf.fit(X_train, Y_train)
    # predict labels in test mask
    predicted_prob    = clf.predict_proba(X_test)
    Y_predicted_prob  = torch.tensor([p[:, 1] for p in predicted_prob]).T
    Y_predicted_label = (Y_predicted_prob > 0.5)*1


                ###############################
                ####### Save Prediction #######
                ###############################
                
    n_classes = len(cfg['labels'])
    prediction = torch.zeros((145,145,145, n_classes))
    prediction.view(-1, n_classes)[test_mask.reshape(-1)  == 1] = Y_predicted_label.float()

                ###############################
                ##### Evaluate Prediction #####
                ###############################
    
    current_correct = torch.eq(Y_predicted_label, torch.tensor(Y_test)) * 1
    total_mistakes = torch.ne(Y_predicted_label, torch.tensor(Y_test))
    false_positives = ((1 - Y_predicted_label) * Y_test)
    false_negatives = (Y_predicted_label * (1-Y_test))
    
    print(current_correct.shape, total_mistakes.shape)
    if prev_correct is None:
        bad_mistakes = torch.zeros(5)   
        bad_positives = torch.zeros(5)   
        bad_negatives = torch.zeros(5)   
    else:
        print("step 3")
        bad_mistakes  = (total_mistakes  * prev_correct).sum(0) / prev_correct.sum(0)
        bad_positives = (false_positives * prev_correct).sum(0)
        bad_negatives = (false_negatives * prev_correct).sum(0)

    
    # Constant for numerical stability
    eps = 1e-5
    # statictics for precision, recall and Dice (f1)
    TP       = (Y_predicted_label * Y_test).sum(axis=0)
    TPplusFP = Y_predicted_label.sum(axis=0)
    TPplusFN = Y_test.sum(axis=0)

    precision = (TP + eps) / (TPplusFP + eps)
    recall    = (TP + eps) / (TPplusFN + eps)
    f1        = (2 * precision * recall + eps) / ( precision + recall  + eps)

    labels = cfg["labels"]
    scores = {}
    for c in range(len(labels)):
        scores[f"{labels[c]}_total_mistakes"] = total_mistakes.sum(0)[c].numpy()
        scores[f"{labels[c]}_bad_mistakes"]   = bad_mistakes[c].numpy()
        scores[f"{labels[c]}_bad_positives"]  = bad_positives[c].numpy()
        scores[f"{labels[c]}_bad_negatives"]  = bad_negatives[c].numpy()

    scores["Avg_bad_mistakes"]  = bad_mistakes[1:].float().mean().numpy()
    scores["Avg_bad_positives"] = bad_positives[1:].float().mean().numpy()
    scores["Avg_bad_negatives"] = bad_negatives[1:].float().mean().numpy()

    return scores, current_correct


def eval_pca(dataset: Dataset, cfg: str, n_components: Iterable = np.arange(0, 50)) -> Union[np.array, List[dict]]:
    
    features_raw = dataset.input.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)
    pca = PCA(n_components=None, random_state=42)
    features_pca = pca.fit_transform(features_raw)
    explained_variance_ratio = pca.explained_variance_ratio_
    scores_list = []

    for i in tqdm(n_components):
        scores, preds = evaluate_RF(dataset, features_pca[:, :i+1], cfg)
        scores_list.append(scores)
        
    return np.cumsum(explained_variance_ratio), scores_list