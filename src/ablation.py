import os, sys
from time import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import copy

import wandb
from tqdm.auto import tqdm
from typing import List, Dict, Iterable, Callable, Generator, Union

from dataset import AEDataset
from pretrainer import PreTrainer
from trainer import WeakSupervisionTrainer
from model import DualBranchAE
from losses import MSELoss
from layer import SegmentationDecoder


os.environ["WANDB_SILENT"] = "True"

def scores2df(scores: Dict[str, float], t: int, ablate: str = 'full', 
              s: int = 2, i: int = 0) -> list:
    rows = []
    for key in scores:
        if 'f1' in key:
            c = key.replace('_tracts', '').rsplit('_', maxsplit=1)[0]
            rows.append({'f1'     : scores[key],
                         'ablate' : ablate,
                         't'      : t,  
                         'class'  : c, 
                         'set'    : s, 
                         'iter'   : i})
            
    return rows


def eval_config_for_set_and_iter(ablate: str, dataset: Dataset, s: int, i: int, 
                                 cfg: dict) -> pd.DataFrame:
    
    # data frame to return
    df = pd.DataFrame(columns=['f1', 'ablate', 't', 'class', 'set', 'iter'])
    # configures ablation study
                # ablation                              # defaults
    encoder     = 'zero' if ablate == 'encoder'     else 'dual'
    thresholds  = 'old'  if ablate == 'thresholds'  else 'learned'
    mse         = False  if ablate == 'mse'         else True
    regularizer = False  if ablate == 'regularizer' else True
    
    print(f'Ablate {ablate}:')
    print(f'Encoder: {encoder} \nThresholds: {thresholds} \nMSE: {mse} \nRegularizer: {regularizer}')

    # init student and teacher models
    teacher = DualBranchAE(encoder    = encoder,
                           decoder    = 'reconstruction',
                           in_size    = 145,
                           thresholds = thresholds).to(cfg['rank'])

    student = DualBranchAE(encoder    = encoder,
                           decoder    = 'segmentation',
                           in_size    = 145,
                           n_classes  = len(cfg['labels']),
                           thresholds = thresholds).to(cfg['rank'])
    # unsupervised pre-training of teacher
    # initial user annotation to track unsupervised feature performance
    dataset.clear_annotation()
    annot = dataset.initial_annotation(seed=i)
    dataset.update_annotation(annot)
    # fit model - unsupervised
    train_loader = DataLoader(dataset, batch_size=cfg['s_batch_size'], shuffle=True, drop_last=False)
    criterion    = MSELoss()
    description  = 'zero_' if ablate == 'encoder' else 'dual_'
    pre_trainer  = PreTrainer(teacher, criterion, train_loader, cfg,
                              n_epochs=cfg['s_n_epochs'], lr=cfg['s_lr'], log=False,
                              description=description+str(i), patience=8)
    try:
        pre_trainer.load_model()
    except:
        pre_trainer.fit()
        
    # for combined loss, store reconstruction decoder
    if ablate != 'mse':
        teacher._modules['decoder_recon'] = teacher._modules.pop('decoder')
    # init untrained segmentation decoder
    teacher.decoder = SegmentationDecoder(n_classes  = len(cfg['labels']),
                                          thresholds = thresholds)
    teacher.to(cfg['rank'])
    teacher_state = copy.deepcopy(teacher.state_dict())
    # transwer teacher parameters to student
    if ablate != 'mse':
        student.load_state_dict(teacher_state)
    else:
        missing_keys, _ = student.load_state_dict(teacher_state, strict=False)
    
    # fit model - weakly supervised
    train_loader = DataLoader(dataset, batch_size=2*cfg['w_batch_size'], shuffle=True, drop_last=False)
    trainer = WeakSupervisionTrainer(mse=mse, regularizer=regularizer)
    
    print("starting self supervision")
    trainer.fit(teacher, train_loader, epochs=cfg['w_n_epochs'], 
                lr=cfg['w_lr'], warm_up=True, log=False, cfg=cfg)
    
    if cfg['log']:
        run = wandb.init(reinit=True, name='log_' + ablate, project=cfg['project'])
        
    for t in range(cfg['num_interactions']):
        # first prediction with unsupervised features, since student has not been updated yet
        # get predictions from student model
        scores, prediction = trainer.evaluate(model=student, dataset=dataset, cfg=cfg)
        # update annotations
        annot = dataset.refinement_annotation(prediction=prediction, seed=i)
        dataset.update_annotation(annotations=annot)
        # update student parameters after inference to establish a one step time lag
        teacher_state = copy.deepcopy(teacher.state_dict())
        if ablate != 'mse':
            student.load_state_dict(teacher_state)
        else:
            missing_keys, _ = student.load_state_dict(teacher_state, strict=False)
        # log RF performance
        if cfg['log']:
            wandb.log({'RF_w'  : scores,
                       'x': t})
            df = df.append(scores2df(scores = scores,
                                     ablate = ablate,
                                     t      = t,  
                                     s      = s, 
                                     i      = i), 
                           ignore_index=True)

        # fine-tune teacher model with new annotations
        trainer.fit(teacher, train_loader, epochs=cfg['w_n_epochs'], 
                    lr=cfg['w_lr'], warm_up=False, log=False, cfg=cfg)
    
    return df


def main():
    # Wrapper function for ablation study
    with open('configs/experiment2_ablation.config', 'r') as config:
        cfg = eval(config.read())
        
    df = pd.DataFrame(columns=['f1', 'ablate', 't', 'class', 'set', 'iter'])
    
    # iterate over set 1 and 2 (named differently internally)
    for s in [3]:
        dataset = AEDataset(cfg, modality='segmentation', normalize=True,
                            set=s, augment=False, to_gpu=True)
        
        # 10 runs for each experiment
        for i in range(10):
            # full
            for ablate in ['full', 'encoder', 'thresholds', 'regularizer', 'mse']:
                tmp = eval_config_for_set_and_iter(ablate  = ablate, 
                                                   dataset = dataset, 
                                                   s       = s, 
                                                   i       = i,
                                                   cfg     = cfg)
                df = df.append(tmp)
                df.to_pickle("tmp/ablation_tmp")
        del dataset
                
    df['f1']  = df['f1'].apply(lambda x: x.item())
    df['set'] = df['set'] - 1
    df.to_pickle('tmp/ablation_set2')

    
if __name__ == '__main__':
    main()