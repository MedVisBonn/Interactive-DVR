#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:02:19 2020

@author: Jonathan Lennartz
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from torch import nn, Tensor
import torch.nn.functional as F
import nibabel as nib
from os.path import join
from pathlib import Path
from time import time
from typing import List, Dict, Iterable, Callable, Generator, Union
import wandb
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
import pandas as pd

def debugging(message):
    print("".center(60, "#"))
    print(message.center(60, "-"))
    print("".center(60, "#"))
    print("\n")
    print("\n")



def save_results(
    results: List[Dict],
    subject_id: str,
    uncertainty_measure: str,
    background_bias: bool,
    feature: bool,
    save_dir: str
):
    # Initialize an empty list to store the structured data
    data = []

    # Iterate through the list to process each result
    for iteration, result in enumerate(results):
        scores = result['scores']
        for key, value in scores.items():
            parts = key.split('_')
            region = parts[0]
            score_type = parts[1] if len(parts) == 2 else parts[1] + "_" + parts[2] # Handle the _tracts case
            data.append({
                'iteration': iteration,
                'region': region,
                'score_type': score_type.replace('_tracts', ''),
                'score': value.item(),  # Convert numpy array to Python scalar
                'subject_id': subject_id,
                'uncertainty_measure': uncertainty_measure, 
                'background_bias': background_bias,
                'feature': feature                        
            })

    # Convert the structured data into a pandas DataFrame
    df = pd.DataFrame(data)
    save_name = f"{subject_id}_{uncertainty_measure}_bb-{background_bias}_{feature}.csv"
    df.to_csv(f'{save_dir}/{save_name}', index=False)



def get_tta_features(
    dataset,
    model,
    verbose,
    var = 0.005,
    n_features = 4
):  
    if verbose:
        print("Extracting TTA features...")
    input_ = copy.deepcopy(dataset.data_in)
    feature_list = []
    for i in range(n_features):
        dataset.data_in = input_ + torch.randn_like(dataset.data_in) * np.sqrt(var)
        extractor = FeatureExtractor(model, layers=['encoder'])
        hooked_results = extractor(dataset)
        features = hooked_results['encoder']
        features = features.permute(0,2,3,1).numpy()
        feature_list.append(features)
    dataset.data_in = input_
    return feature_list


def get_features(
    model: nn.Module, 
    dataset: Dataset,
    feature: str = 'default',
    verbose: bool = False
):  
    if verbose:
        print("Extracting features...")
    f_layer = 'encoder'
    extractor = FeatureExtractor(model, layers=[f_layer])
    hooked_results = extractor(dataset)
    features = hooked_results[f_layer]
    features = [features.permute(0,2,3,1).numpy()]
    if feature == 'tta':
        features = features + get_tta_features(dataset, model, verbose)
    if verbose:
        print("Done.\n")
    return features




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
    
### Background Bias
def add_background_bias(
    prediction: Tensor,
    anomaly_score_map: Tensor,
    background_class: Tensor,
    threshold: float
):
    # Check inputs
    assert prediction.shape[1:] == anomaly_score_map.shape
    assert len(prediction.shape) == 4
    assert len(background_class) == prediction.shape[0], f"{background_class.shape} != {prediction.shape[0]}"
    assert threshold >= 0 and threshold <= 1

    # Add background bias
    prediction[:, anomaly_score_map>threshold] = background_class[:, None].float()

    return prediction


### Uncertainty measures
def calc_binary_entropy(prob):
    eps = 1e-7
    
    #Edge Cases
    prob = torch.clamp(prob, eps, 1-eps)
    
    return -(prob*torch.log2(prob) + (1-prob)*torch.log2(1-prob))


def uncertainty_entropy(
    Y_predicted_prob, 
    n_classes, 
    test_mask
):
    
    entropy = calc_binary_entropy(Y_predicted_prob)
    entropy_map = torch.zeros((145,145,145, n_classes))
    entropy_map[test_mask  == 1] = entropy.float()
    mean_entropy_map = entropy_map.mean(axis=-1)
    return mean_entropy_map, entropy_map


def uncertainty_sd(
    train_label, 
    test_mask, 
    n_classes
):
    train_label_tensor = torch.from_numpy(train_label)
    annotated_voxels = torch.any(train_label_tensor, dim=-1)
    # per class
    sd_map_per_class = torch.zeros((n_classes, 145,145,145))
    for i in range(n_classes):
        train_label_i = train_label_tensor[:,:,:,i].bool()
        sd = torch.tensor(distance_transform_edt(~train_label_i))
        sd_map_per_class[i, test_mask == 1] = sd[test_mask == 1].float()  
    sd_map_per_class[:, annotated_voxels] = 0 # all values of annotated voxels should be 0
    # all classes
    spatial_distance_map, _ = torch.min(sd_map_per_class, dim=0)
    
    return spatial_distance_map, sd_map_per_class.permute(1,2,3,0)


def uncertainty_fd(
    train_label, 
    features, 
    test_mask, 
    n_classes,
    class_wise: str = False
):
    
    train_label_tensor = torch.from_numpy(train_label)
    annotated_voxels = torch.any(train_label_tensor, dim=-1)
    brain_mask_tensor = torch.from_numpy(test_mask == 1)
    
    def compute_anomaly_scores(annotated_features, mask):
        iforest = IsolationForest(n_estimators=100, random_state=0, n_jobs=-1).fit(annotated_features)
        anomaly_scores = iforest.decision_function(features[mask].reshape(-1, 44))
        return torch.from_numpy(anomaly_scores).float()
        # anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
        # return torch.from_numpy(1 - anomaly_scores).float()
    
    

    # all classes
    annotated_features = features[annotated_voxels].reshape(-1, 44)
    brain_na_mask = brain_mask_tensor & ~annotated_voxels
    anomaly_scores_map = torch.zeros((145,145,145))
    anomaly_scores_map[brain_na_mask] = compute_anomaly_scores(annotated_features, brain_na_mask)
    min_score = anomaly_scores_map.min()
    max_score = anomaly_scores_map.max()

    anomaly_scores_map = (anomaly_scores_map - min_score) / (max_score - min_score)
    anomaly_scores_map = 1 - anomaly_scores_map
    t = (0 - min_score) / (max_score - min_score)
    t = 1 - t
    # anomaly_scores_map -= anomaly_scores_map.min()
    # anomaly_scores_map /= anomaly_scores_map.max()

    # per class
    fd_map_per_class = torch.zeros((n_classes, 145,145,145))
    if class_wise:
        for i in range(n_classes):
            train_label_i = train_label_tensor[:,:,:,i].bool()
            annotated_features = features[train_label_i].reshape(-1, 44)
            brain_na_mask = brain_mask_tensor & ~train_label_i
            fd_map_per_class[i, brain_na_mask == 1] = compute_anomaly_scores(annotated_features, brain_na_mask)
        fd_map_per_class[:, annotated_voxels] = 0 # all values of annotated voxels should be 0
        # fd_map_per_class -= fd_map_per_class.amin(dim=(1,2,3), keepdim=True)
        # fd_map_per_class /= fd_map_per_class.amax(dim=(1,2,3), keepdim=True)

    return anomaly_scores_map, fd_map_per_class.permute(1,2,3,0), t



def get_scores(
    pred, 
    gt,
    cfg
):
    # Constant for numerical stability
    eps = 1e-5
    # statictics for precision, recall and Dice (f1)
    TP       = (pred * gt).sum(axis=1)
    TPplusFP = pred.sum(axis=1)
    TPplusFN = gt.sum(axis=1)

    precision = (TP + eps) / (TPplusFP + eps)
    recall    = (TP + eps) / (TPplusFN + eps)
    f1        = (2 * precision * recall + eps) / ( precision + recall  + eps)
    
    labels = cfg['data']["labels"]
    scores = {}
    for c in range(len(labels)):
        scores[f"{labels[c]}_precision"] = precision[c].numpy()
        scores[f"{labels[c]}_recall"] = recall[c].numpy()
        scores[f"{labels[c]}_f1"] = f1[c].numpy()

    scores["Avg_prec_tracts"] = precision[1:].mean().numpy()
    scores["Avg_recall_tracts"] = recall[1:].mean().numpy()
    scores["Avg_f1_tracts"] = f1[1:].mean().numpy()

    return scores



def evaluate_RF(
    dataset: Dataset, 
    features: Tensor, 
    cfg: Dict[str, str], 
    uncertainty_measures: List[str], 
    tta=False
)-> Union[Dict[str, float], Tensor]:

                ###############################
                ##### TRAIN RANDOM FOREST #####
                ###############################
                
    train_mask  = dataset.weight.detach().cpu().squeeze().numpy()    #.permute(0,2,3,1).repeat(1,1,1,44).numpy()
    test_mask   = dataset.brain_mask.detach().cpu().numpy()#.unsqueeze(3).repeat(1,1,1,44).numpy()
    train_label = dataset.annotations.detach().cpu().permute(1,2,3,0).numpy()
    test_label  = dataset.label.detach().cpu().permute(1,2,3,0).numpy()
    
    if cfg.feature=='tta':
        f = np.stack(features[1:], axis=0)  # (n_tta, 145, 145, 145, 44)
        num_tta = len(features) - 1
        train_m = np.repeat(train_mask[np.newaxis, ...], num_tta, axis=0)  # (n_tta, 145, 145, 145)
        test_m = np.repeat(test_mask[np.newaxis, ...], num_tta, axis=0)  # (n_tta, 145, 145, 145)
        train_l = np.repeat(train_label[np.newaxis, ...], num_tta, axis=0)  # (n_tta, 145, 145, 145, 5)

        X_train = f.reshape((-1, f.shape[-1]))[train_m.reshape(-1) == 1]    # (n_tta*train_voxels, 44)
        X_test = f.reshape((-1, f.shape[-1]))[test_m.reshape(-1) == 1]    # (n_tta*844350, 44)
        Y_train = train_l.reshape((-1, train_l.shape[-1]))[train_m.reshape(-1) == 1]   # (n_tta*train_voxels, 5)
    
    else:
        # Input - Mask voxels that are not labelled before flattening the input
        X_train = features[0].reshape((-1, features[0].shape[-1]))[train_mask.reshape(-1) == 1]
        X_test  = features[0].reshape((-1, features[0].shape[-1]))[test_mask.reshape(-1)  == 1]
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
    n_classes = len(cfg['data']['labels'])
    if cfg.feature=='tta':
        Y_predicted_prob = Y_predicted_prob.reshape((num_tta, Y_test.shape[0], n_classes)).mean(axis=0)
    Y_predicted_label = (Y_predicted_prob > 0.5)*1


                ###############################
                ####### Save Prediction #######
                ###############################
                
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

    uncertainty_maps = {}
    uncertainty_per_class_maps = {}

    for measure in uncertainty_measures:
        t = None
        match measure:
            case 'ground-truth':
                continue
            case 'random':
                break
            case 'entropy':
                uncertainty_map, uncertainty_per_class = uncertainty_entropy(Y_predicted_prob, n_classes, test_mask)
            case 'spatial-distance':
                uncertainty_map, uncertainty_per_class = uncertainty_sd(train_label, test_mask, n_classes)
            case 'feature-distance':
                uncertainty_map, uncertainty_per_class, t = uncertainty_fd(train_label, features[0], test_mask, n_classes)
            case _:
                raise ValueError(f"Uncertainty measure {measure} not implemented")

        uncertainty_maps[measure] = uncertainty_map
        uncertainty_per_class_maps[measure] = uncertainty_per_class.permute(3,0,1,2)

    #TODO: for measure in novelty_scores:

    labels = cfg['data']["labels"]
    scores = {}
    for c in range(len(labels)):
        scores[f"{labels[c]}_precision"] = precision[c].numpy()
        scores[f"{labels[c]}_recall"] = recall[c].numpy()
        scores[f"{labels[c]}_f1"] = f1[c].numpy()

    scores["Avg_prec_tracts"] = precision[1:].mean().numpy()
    scores["Avg_recall_tracts"] = recall[1:].mean().numpy()
    scores["Avg_f1_tracts"] = f1[1:].mean().numpy()
    
    
    return scores, prediction.permute(3,0,1,2), uncertainty_maps, uncertainty_per_class_maps, t



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


# def evaluate_RF(dataset: Dataset, features: Tensor, cfg: Dict[str, str]) \
#                 -> Union[Dict[str, float], Tensor]:

#                 ###############################
#                 ##### TRAIN RANDOM FOREST #####
#                 ###############################
                
#     train_mask  = dataset.weight.detach().cpu().squeeze().numpy()    #.permute(0,2,3,1).repeat(1,1,1,44).numpy()
#     test_mask   = dataset.brain_mask.detach().cpu().numpy()#.unsqueeze(3).repeat(1,1,1,44).numpy()
#     train_label = dataset.annotations.detach().cpu().permute(1,2,3,0).numpy()
#     test_label  = dataset.label.detach().cpu().permute(1,2,3,0).numpy()
    
#     # Input - Mask voxels that are not labelled before flattening the input
#     X_train = features.reshape((-1, features.shape[-1]))[train_mask.reshape(-1) == 1]
#     X_test  = features.reshape((-1, features.shape[-1]))[test_mask.reshape(-1)  == 1]
#     # Target - Same as above. Mask before flattening
#     Y_train = train_label.reshape((-1, train_label.shape[-1]))[train_mask.reshape(-1) == 1]
#     Y_test  = test_label.reshape((-1,  train_label.shape[-1]))[test_mask.reshape(-1)  == 1]

#     # Init Random Forest Classifier
#     clf = RandomForestClassifier(n_estimators=100,
#                          bootstrap=True,
#                          oob_score=True,
#                          random_state=0,
#                          n_jobs=-1,
#                          max_features="auto",
#                          class_weight="balanced",
#                          max_depth=None,
#                          min_samples_leaf=cfg["min_samples_leaf"])

#     # Train
#     clf.fit(X_train, Y_train)
#     # predict labels in test mask
#     predicted_prob    = clf.predict_proba(X_test)
#     Y_predicted_prob  = torch.tensor([p[:, 1] for p in predicted_prob]).T
#     Y_predicted_label = (Y_predicted_prob > 0.5)*1


#                 ###############################
#                 ####### Save Prediction #######
#                 ###############################
                
#     n_classes = len(cfg['labels'])
#     prediction = torch.zeros((145,145,145, n_classes))
#     prediction.view(-1, n_classes)[test_mask.reshape(-1)  == 1] = Y_predicted_label.float()

#                 ###############################
#                 ##### Evaluate Prediction #####
#                 ###############################

#     # Constant for numerical stability
#     eps = 1e-5
#     # statictics for precision, recall and Dice (f1)
#     TP       = (Y_predicted_label * Y_test).sum(axis=0)
#     TPplusFP = Y_predicted_label.sum(axis=0)
#     TPplusFN = Y_test.sum(axis=0)

#     precision = (TP + eps) / (TPplusFP + eps)
#     recall    = (TP + eps) / (TPplusFN + eps)
#     f1        = (2 * precision * recall + eps) / ( precision + recall  + eps)

#     labels = cfg["labels"]
#     scores = {}
#     for c in range(len(labels)):
#         scores[f"{labels[c]}_precision"] = precision[c].numpy()
#         scores[f"{labels[c]}_recall"] = recall[c].numpy()
#         scores[f"{labels[c]}_f1"] = f1[c].numpy()

#     scores["Avg_prec_tracts"] = precision[1:].mean().numpy()
#     scores["Avg_recall_tracts"] = recall[1:].mean().numpy()
#     scores["Avg_f1_tracts"] = f1[1:].mean().numpy()
    
    

#     return scores, prediction.permute(3,0,1,2)


# def evaluate_RF_tmp(dataset: Dataset, features: Tensor, prev_correct: Tensor,
#                     cfg: Dict[str, str]) -> Union[Dict[str, float], Tensor]:

#                 ###############################
#                 ##### TRAIN RANDOM FOREST #####
#                 ###############################
                
#     train_mask  = dataset.weight.detach().cpu().squeeze().numpy()    #.permute(0,2,3,1).repeat(1,1,1,44).numpy()
#     test_mask   = dataset.brain_mask.detach().cpu().numpy()#.unsqueeze(3).repeat(1,1,1,44).numpy()
#     train_label = dataset.annotations.detach().cpu().permute(1,2,3,0).numpy()
#     test_label  = dataset.label.detach().cpu().permute(1,2,3,0).numpy()
    
#     # Input - Mask voxels that are not labelled before flattening the input
#     X_train = features.reshape((-1, features.shape[-1]))[train_mask.reshape(-1) == 1]
#     X_test  = features.reshape((-1, features.shape[-1]))[test_mask.reshape(-1)  == 1]
#     # Target - Same as above. Mask before flattening
#     Y_train = train_label.reshape((-1, train_label.shape[-1]))[train_mask.reshape(-1) == 1]
#     Y_test  = test_label.reshape((-1,  train_label.shape[-1]))[test_mask.reshape(-1)  == 1]

#     # Init Random Forest Classifier
#     clf = RandomForestClassifier(n_estimators=100,
#                          bootstrap=True,
#                          oob_score=True,
#                          random_state=0,
#                          n_jobs=-1,
#                          max_features="auto",
#                          class_weight="balanced",
#                          max_depth=None,
#                          min_samples_leaf=cfg["min_samples_leaf"])

#     # Train
#     clf.fit(X_train, Y_train)
#     # predict labels in test mask
#     predicted_prob    = clf.predict_proba(X_test)
#     Y_predicted_prob  = torch.tensor([p[:, 1] for p in predicted_prob]).T
#     Y_predicted_label = (Y_predicted_prob > 0.5)*1


#                 ###############################
#                 ####### Save Prediction #######
#                 ###############################
                
#     n_classes = len(cfg['labels'])
#     prediction = torch.zeros((145,145,145, n_classes))
#     prediction.view(-1, n_classes)[test_mask.reshape(-1)  == 1] = Y_predicted_label.float()

#                 ###############################
#                 ##### Evaluate Prediction #####
#                 ###############################
    
#     current_correct = torch.eq(Y_predicted_label, torch.tensor(Y_test)) * 1
#     total_mistakes = torch.ne(Y_predicted_label, torch.tensor(Y_test))
#     false_positives = ((1 - Y_predicted_label) * Y_test)
#     false_negatives = (Y_predicted_label * (1-Y_test))
    
#     print(current_correct.shape, total_mistakes.shape)
#     if prev_correct is None:
#         bad_mistakes = torch.zeros(5)   
#         bad_positives = torch.zeros(5)   
#         bad_negatives = torch.zeros(5)   
#     else:
#         print("step 3")
#         bad_mistakes  = (total_mistakes  * prev_correct).sum(0) / prev_correct.sum(0)
#         bad_positives = (false_positives * prev_correct).sum(0)
#         bad_negatives = (false_negatives * prev_correct).sum(0)

    
#     # Constant for numerical stability
#     eps = 1e-5
#     # statictics for precision, recall and Dice (f1)
#     TP       = (Y_predicted_label * Y_test).sum(axis=0)
#     TPplusFP = Y_predicted_label.sum(axis=0)
#     TPplusFN = Y_test.sum(axis=0)

#     precision = (TP + eps) / (TPplusFP + eps)
#     recall    = (TP + eps) / (TPplusFN + eps)
#     f1        = (2 * precision * recall + eps) / ( precision + recall  + eps)

#     labels = cfg["labels"]
#     scores = {}
#     for c in range(len(labels)):
#         scores[f"{labels[c]}_total_mistakes"] = total_mistakes.sum(0)[c].numpy()
#         scores[f"{labels[c]}_bad_mistakes"]   = bad_mistakes[c].numpy()
#         scores[f"{labels[c]}_bad_positives"]  = bad_positives[c].numpy()
#         scores[f"{labels[c]}_bad_negatives"]  = bad_negatives[c].numpy()

#     scores["Avg_bad_mistakes"]  = bad_mistakes[1:].float().mean().numpy()
#     scores["Avg_bad_positives"] = bad_positives[1:].float().mean().numpy()
#     scores["Avg_bad_negatives"] = bad_negatives[1:].float().mean().numpy()

#     return scores, current_correct


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


def simulate_user_interaction(
    dataset : Dataset, 
    features: Tensor,
    uncertainty_measures: List[str],
    cfg,
    verbose: bool = False
):

    results = []

    # results of initial annotation
    scores, prediction, uncertainty_maps, uncertainty_per_class_maps, t = evaluate_RF(
        dataset=dataset, 
        features=features, 
        cfg=cfg,
        uncertainty_measures=uncertainty_measures
    )

    # add background bias
    if cfg.background_bias:
        background_class = torch.zeros(len(cfg.data.labels))
        background_class[0] = 1
        prediction = add_background_bias(
            prediction=prediction,
            anomaly_score_map=uncertainty_maps['feature-distance'],
            background_class=background_class,
            threshold=t
        )

        scores = get_scores(
            pred=prediction.flatten(1),
            gt=dataset.label.detach().cpu().flatten(1),
            cfg=cfg
        )


    results.append(
        {
            'scores': scores,
            # 'prediction': prediction.clone(),
            # 'uncertainty_maps': uncertainty_maps,
            # 'uncertainty_per_class_maps': uncertainty_per_class_maps,
            # 'num_annotations' : dataset.annotations.detach().cpu().sum().item(),
            # 'num_annotated_voxels' : dataset.annotations.detach().cpu().any(dim=0).sum().item()
        }
    )

    # print(results[0]['scores'])
    # print(results[0]['num_annotations'])
    # print(results[0]['num_annotated_voxels'])

    # cyclic process of user interaction
    for i in tqdm(range(cfg.num_interactions), desc='User interaction', unit='iteration'):


        u_annots, _ = dataset.user.refinement_annotation(
            prediction=prediction,
            annotation_mask=dataset.annotations.detach().cpu(),
            uncertainty_map=uncertainty_per_class_maps['entropy'],
            n_samples=200,
            mode='per_class',
            seed=42,
            inverse_class_freq=False
        )
        dataset.update_annotation(u_annots)
        
        scores, prediction, uncertainty_maps, uncertainty_per_class_maps, t = evaluate_RF(
            dataset=dataset, 
            features=features, 
            cfg=cfg,
            uncertainty_measures=uncertainty_measures
        )

        # add background bias
        if cfg.background_bias:
            background_class = torch.zeros(len(cfg.data.labels))
            background_class[0] = 1
            prediction = add_background_bias(
                prediction=prediction,
                anomaly_score_map=uncertainty_maps['feature-distance'],
                background_class=background_class,
                threshold=t
            )
            scores = get_scores(
                pred=prediction.flatten(1),
                gt=dataset.label.detach().cpu().flatten(1),
                cfg=cfg
            )

        results.append(
            {
                'scores': scores,
                # 'prediction': prediction.clone(),
                # 'uncertainty_maps': uncertainty_maps,
                # 'uncertainty_per_class_maps': uncertainty_per_class_maps,
                # 'num_annotations' : dataset.annotations.detach().cpu().sum().item(),
                # 'num_annotated_voxels' : dataset.annotations.detach().cpu().any(dim=0).sum().item()
            }
        )

    return results
    # for r in results:
    #     print(r['scores']['Avg_f1_tracts']) 