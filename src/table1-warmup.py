import os, sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from time import time
import wandb
from tqdm.auto import tqdm
from typing import List, Dict, Iterable, Callable, Generator, Union

from dataset import AEDataset
from utils import evaluate_RF, eval_pca
from pretrainer import PreTrainer
from trainer import WeakSupervisionTrainer
from model import DualBranchAE
from losses import MSELoss
from layer import SegmentationDecoder, DualLinkEncoder



os.environ["WANDB_SILENT"] = "True"


def scores2df(scores: Dict[str, float], m: str = 'pre_train', 
              s: int = 2, i: int = 0) -> list:
    rows = []
    for key in scores:
        if 'f1' in key:
            c = key.replace('_tracts', '').rsplit('_', maxsplit=1)[0]
            rows.append({'f1'     : scores[key], 
                         'method' : m, 
                         'class'  : c, 
                         'set'    : s, 
                         'iter'   : i})
            
    return rows


def eval_set_for_iter(dataset: Dataset, s: int, i: int, cfg: dict) -> pd.DataFrame:
    
    print("\n")
    print(f"Iteration {i} - Set {s}".center(40, "-"))
    
    # data frame
    df = pd.DataFrame(columns=['f1', 'method', 'class', 'set', 'iter'])
    
    # CNN
    print("Pre-Trained:", end=' ')
    dataset.clear_annotation()
    annot = dataset.initial_annotation(seed=i)
    dataset.update_annotation(annot)
    train_loader = DataLoader(dataset, batch_size=cfg['s_batch_size'], 
                              shuffle=True, drop_last=False)

    model = DualBranchAE(encoder    = 'zero',
                         decoder    = 'reconstruction',
                         in_size    = 145,
                         n_classes  = len(cfg['labels']),
                         thresholds = 'learned').to(cfg['rank'])

    criterion = MSELoss()
    # description = f'Table1_{str(i)}_set{str(set)}'
    description = f'zero_{str(i)}'
    pre_trainer = PreTrainer(model, criterion, train_loader, cfg, n_epochs=cfg['s_n_epochs'], 
                             lr=cfg['s_lr'], log=True, description=description,
                             patience=8, es_mode='none')
    #try:
    pre_trainer.load_model()
    #except:
    #    pre_trainer.fit()
    
    scores, predictions = pre_trainer.evaluate_rf()
    df = df.append(scores2df(scores, m='pre_trained', s=s, i=i), ignore_index=True)
    print(u'\u2713')

    train_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=False)
    
    #######################
    
    pre_trained_encoder_state = model.encoder.state_dict()
    model.encoder.cpu()
    model.encoder = DualLinkEncoder(145)
    model.load_encoder_state(pre_trained_encoder_state)
    # for combined loss, store reconstruction decoder
    model._modules['decoder_recon'] = model._modules.pop('decoder')
    # init untrained segmentation decoder
    model.decoder = SegmentationDecoder(n_classes  = len(cfg['labels']),
                                        thresholds = 'learned')
    model.to(cfg['rank'])
    
    seg_trainer = WeakSupervisionTrainer(mse=False, regularizer=True)
    seg_trainer.fit(model, train_loader, epochs=cfg['w_n_epochs'], 
                    lr=cfg['w_lr'], warm_up=True, extended_warmup=False,
                    cfg=cfg)
    
    print("Re-Trained:", end=' ')
    scores, predictions = seg_trainer.evaluate(model,
                                               dataset,
                                               cfg)
    
    df = df.append(scores2df(scores, m='warmup', s=s, i=i), ignore_index=True)
    print('\u2713')
    
    #############################
    
    model.encoder.cpu()
    model.encoder = DualLinkEncoder(145)
    model.load_encoder_state(pre_trained_encoder_state)
    # init untrained segmentation decoder
    model.decoder = SegmentationDecoder(n_classes    = len(cfg['labels']),
                                          thresholds = 'learned')
    model.to(cfg['rank'])
    
    seg_trainer = WeakSupervisionTrainer(mse=False, regularizer=True)
    seg_trainer.fit(model, train_loader, epochs=cfg['w_n_epochs'], 
                    lr=cfg['w_lr'], warm_up=True, extended_warmup=True,
                    cfg=cfg)
    
    print("Re-Trained:", end=' ')
    scores, predictions = seg_trainer.evaluate(model,
                                               dataset,
                                               cfg)
    
    df = df.append(scores2df(scores, m='extended_warmup', s=s, i=i), ignore_index=True)
    print('\u2713')
    
    return df



def main():
    start = time()
    with open('configs/experiment1_table.config', 'r') as config:
        cfg = eval(config.read())
        
    df = pd.DataFrame(columns=['f1', 'method', 'class', 'set', 'iter'])
    
    # iterate over set 1 and 2 (named differently internally)
    for s in [2, 3]:
        dataset = AEDataset(cfg, modality='segmentation', normalize=True,
                            set=s, augment=False, to_gpu=True)

        # 10 runs for each experiment
        for i in range(10):
            tmp = eval_set_for_iter(dataset=dataset, s=s, i=i, cfg=cfg)
            df  = df.append(tmp, ignore_index=True)
            df.to_pickle("tmp/tmp_table1_warmup_data")
    
    # final clearning for readability
    df['f1']  = df['f1'].apply(lambda x: x.item())
    df['set'] = df['set'] - 1
    df.to_pickle("../table1_warmup_data")
    
    if cfg['log']:
        try:
            wandb.alert(
                title="Table 1 Done", 
                text=f"All methods for table 1 came to an end. It took {(time() - start)/60:.2f} minutes."
            )
        except Exception:
            print("Could not make alert. Error:")
            print(Exception)

if __name__ == '__main__':
    main()