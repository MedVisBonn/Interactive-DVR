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

def scores2df(scores: Dict[str, float], t: int, supervision: bool = True, 
              s: int = 2, i: int = 0) -> list:
    rows = []
    for key in scores:
        if 'f1' in key:
            c = key.replace('_tracts', '').rsplit('_', maxsplit=1)[0]
            rows.append({'f1'          : scores[key],
                         'supervision' : supervision,
                         't'           : t,  
                         'class'       : c, 
                         'set'         : s, 
                         'iter'        : i})
            
    return rows


def eval_config_for_set_and_iter(supervision: bool, dataset: Dataset, s: int, i: int, 
                                 cfg: dict) -> pd.DataFrame:
    
    # data frame to return
    df = pd.DataFrame(columns=['f1', 'supervision', 't', 'class', 'set', 'iter'])
    # configures model
    encoder     = 'dual'
    thresholds  = 'learned'
    mse         = True
    regularizer = True
    
    print(f'Supervision {supervision}:')

    # init model
    model = DualBranchAE(encoder    = encoder,
                         decoder    = 'reconstruction',
                         in_size    = 145,
                         thresholds = thresholds).to(cfg['rank'])
    
    # unsupervised pre-training
    # initial user annotation to track unsupervised feature performance
    dataset.clear_annotation()
    annot = dataset.initial_annotation(seed=i)
    dataset.update_annotation(annot)
    # fit model - unsupervised
    train_loader = DataLoader(dataset, batch_size=cfg['s_batch_size'], shuffle=True, drop_last=False)
    criterion    = MSELoss()
    description  = 'dual_'
    pre_trainer  = PreTrainer(model, criterion, train_loader, cfg,
                              n_epochs=cfg['s_n_epochs'], lr=cfg['s_lr'], log=False,
                              description=description+str(i), patience=8)
    try:
        pre_trainer.load_model()
    except:
        pre_trainer.fit()
        
    # for combined loss, store reconstruction decoder
    model._modules['decoder_recon'] = model._modules.pop('decoder')
    # init untrained segmentation decoder
    model.decoder = SegmentationDecoder(n_classes    = len(cfg['labels']),
                                          thresholds = thresholds)
    model.to(cfg['rank'])
    
    # fit model - weakly supervised
    train_loader = DataLoader(dataset, batch_size=cfg['w_batch_size'], shuffle=True, drop_last=False)
    trainer      = WeakSupervisionTrainer(mse=mse, regularizer=regularizer)
    
    if cfg['log']:
        run = wandb.init(reinit=True, name='log_' + str{supervision}, project=cfg['project'])
        
    for t in range(cfg['num_interactions']):
        # first prediction with unsupervised features, since student has not been updated yet
        # get predictions from student model
        scores, prediction = trainer.evaluate(model=model, dataset=dataset, cfg=cfg)
        # fine-tune model with annotations from previous iteration
        if supervision:
            warm_up = True if t == 0 else False
            trainer.fit(model, train_loader, epochs=cfg['w_n_epochs'], 
                        lr=cfg['w_lr'], warm_up=warm_up, log=False, cfg=cfg)
        # update annotations with predictions from pre-refinement model
        annot = dataset.refinement_annotation(prediction=prediction, seed=i)
        dataset.update_annotation(annotations=annot)
        # log RF performance
        if cfg['log']:
            wandb.log({'RF_w'  : scores,
                       'x': t})
            df = df.append(scores2df(scores      = scores,
                                     supervision = supervision,
                                     t           = t,  
                                     s           = s, 
                                     i           = i), 
                           ignore_index=True)
    return df


def main():
    # Wrapper function for supervision experiment
    with open('configs/experiment3_supervision.config', 'r') as config:
        cfg = eval(config.read())
    
    df = pd.DataFrame(columns=['f1', 'supervision', 't', 'class', 'set', 'iter'])
    # iterate over set 1 and 2 (named differently internally)
    for s in [3]:
        dataset = AEDataset(cfg, modality='segmentation', normalize=True,
                            set=s, augment=False, to_gpu=True)
        
        # 10 runs for each experiment
        for i in range(10):
            # full
            for supervison in [True, False]:
                tmp = eval_config_for_set_and_iter(supervison = supervison, 
                                                   dataset    = dataset, 
                                                   s          = s, 
                                                   i          = i,
                                                   cfg        = cfg)
                df = df.append(tmp)
                df.to_pickle("tmp/supervision_tmp")
        del dataset
                
    df['f1']  = df['f1'].apply(lambda x: x.item())
    df['set'] = df['set'] - 1
    df.to_pickle('tmp/supervision')
    
    if cfg['log']:
        wandb.alert(
            title="Supervision Done", 
            text="Runs with and without supervision came to an end"
        )

    
if __name__ == '__main__':
    main()