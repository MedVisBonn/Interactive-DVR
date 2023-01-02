import os, sys
from time import time
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import copy
from time import time
import wandb
from tqdm.auto import tqdm
from typing import List, Dict, Iterable, Callable, Generator, Union

from dataset import AEDataset
from pretrainer import PreTrainer
from trainer import WeakSupervisionTrainer
from model import DualBranchAE
from losses import MSELoss
from layer import SegmentationDecoder, DualLinkEncoder


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
    encoder     = 'zero' if (ablate == 'encoder'     or ablate == 'baseline') else 'dual'
    thresholds  = 'old'  if (ablate == 'thresholds'  or ablate == 'decoder')  else 'learned'
    regularizer = False  if (ablate == 'regularizer' or ablate == 'decoder')  else True
    mse         = True
    
    print(f'Ablate {ablate}:')
    print(f'Encoder: {encoder} \nThresholds: {thresholds} \nRegularizer: {regularizer} \nMSE: {mse}')

    # init model
    model = DualBranchAE(encoder    = 'zero',
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
    description  = 'zero_'
    description += str(i)
    pre_trainer  = PreTrainer(model, criterion, train_loader, cfg,
                              n_epochs=cfg['s_n_epochs'], lr=cfg['s_lr'], log=False,
                              description=description, patience=8, es_mode='none')    
    try:
        pre_trainer.load_model()
    except:
        print("fitting model. Intended?")
        pre_trainer.fit()
    
    if encoder == 'dual':
        # override encoder to get cross connections and load state dict
        pre_trained_state = model.state_dict()
        model.encoder.cpu()
        model.encoder = DualLinkEncoder(145)
        model.load_encoder_state(pre_trained_state)
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
        run = wandb.init(reinit=True, name='log_' + ablate, project=cfg['project'])
        
    for t in range(cfg['num_interactions']):
        # first prediction with unsupervised features, since student has not been updated yet
        # get predictions from student model
        # in first iteration, use encoder without cross connections, since they haven't been
        # learned yet
        # fine-tune model with annotations from previous iteration
        if ablate != 'baseline':
            warm_up = True if t == 0 else False
            n_epochs = cfg['w_n_epochs']
            trainer.fit(model, train_loader, epochs=n_epochs, 
                        lr=cfg['w_lr'], warm_up=warm_up, log=False, cfg=cfg)
        
        scores, prediction = trainer.evaluate(model=model, dataset=dataset, cfg=cfg)
        # update annotations with predictions from pre-refinement model
        annot = dataset.refinement_annotation(prediction=prediction, seed=i)
        dataset.update_annotation(annotations=annot)
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
    return df


def main():
    # Wrapper function for ablation study
    start = time()
    with open('configs/experiment2_ablation.config', 'r') as config:
        cfg = eval(config.read())
    
    df = pd.DataFrame(columns=['f1', 'ablate', 't', 'class', 'set', 'iter'])
    # iterate over set 1 and 2 (named differently internally)
    for s in [2, 3]:
        dataset = AEDataset(cfg, modality='segmentation', normalize=True,
                            set=s, augment=False, to_gpu=True)
        
        # 10 runs for each experiment
        for i in range(10):
            # full
            for ablate in ['full', 'baseline']:
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
    df.to_pickle('../ablation_noLag_mse')
    
    if cfg['log']:
        wandb.alert(
            title="Ablation Done", 
            text=f"All ablations came to an end. It took {(time() - start)/60:.2f} minutes."
            )        

    
if __name__ == '__main__':
    main()