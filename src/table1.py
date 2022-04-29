import os, sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import wandb
from tqdm.auto import tqdm
from typing import List, Dict, Iterable, Callable, Generator, Union

from dataset import AEDataset
from utils import evaluate_RF, eval_pca
from pretrainer import PreTrainer
from trainer import WeakSupervisionTrainer
from model import DualBranchAE
from losses import MSELoss

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


def eval_set_for_iter(datasets: List[Dataset], s: int, i: int, cfg: dict) -> pd.DataFrame:
    
    print("\n")
    print(f"Iteration {i} - Set {s}".center(40, "-"))
    
    # data frame
    df = pd.DataFrame(columns=['f1', 'method', 'class', 'set', 'iter'])

    # raw
    print("Raw:", end=' ')
    datasets['raw'].clear_annotation()
    annot = datasets['raw'].initial_annotation(seed=i)
    datasets['raw'].update_annotation(annot)

    features_raw = datasets['raw'].input.permute(0, 2, 3, 1).flatten(start_dim=0, end_dim=2)
    scores, preds = evaluate_RF(datasets['raw'], features_raw, cfg)
    df = df.append(scores2df(scores, m='raw', s=s, i=i), ignore_index=True)
    print(u'\u2713')
    
    # pca
    datasets['pca'].clear_annotation()
    annot = datasets['pca'].initial_annotation(seed=i)
    datasets['pca'].update_annotation(annot)
    
    cumvar, scores_list = eval_pca(datasets['pca'], n_components=[10], cfg=cfg)
    avrg_f1 = [s['Avg_f1_tracts'] for s in scores_list]
    idx = np.argmax(avrg_f1)
    df  = df.append(scores2df(scores_list[idx], m='pca', s=s, i=i), ignore_index=True)
    print("PCA:", end=' ')
    print(u'\u2713')
    
    # CNN
    print("Un-Trained:", end=' ')
    datasets['cnn'].clear_annotation()
    annot = datasets['cnn'].initial_annotation(seed=i)
    datasets['cnn'].update_annotation(annot)
    train_loader = DataLoader(datasets['cnn'], batch_size=16, shuffle=True, drop_last=False)

    model = DualBranchAE(encoder    = 'dual',
                         decoder    = 'reconstruction',
                         in_size    = 145,
                         n_classes  = len(cfg['labels']),
                         thresholds = 'learned').to(cfg['rank'])

    criterion = MSELoss()
    description = f'Table1_{str(i)}_set{str(set)}' 
    pre_trainer = PreTrainer(model, criterion, train_loader, cfg, n_epochs=10, lr=5e-4, log=True, description=description, patience=8)
    
    scores, predictions = pre_trainer.evaluate_rf()
    df = df.append(scores2df(scores, m='un_trained', s=s, i=i), ignore_index=True)
    print(u'\u2713')
    
    pre_trainer.fit()
    #try:
    #    pre_trainer.load_model()
    #except:
    #    pre_trainer.fit()
    
    print("Pre-Trained:", end=' ')   
    scores, predictions = pre_trainer.evaluate_rf()
    df = df.append(scores2df(scores, m='pre_trained', s=s, i=i), ignore_index=True)
    del model
    print(u'\u2713')
    
    model = DualBranchAE(encoder    = 'dual',
                         decoder    = 'segmentation',
                         in_size    = 145,
                         n_classes  = len(cfg['labels']),
                         thresholds = 'learned').to(cfg['rank'])
    train_loader = DataLoader(datasets['cnn'], batch_size=8, shuffle=True, drop_last=False)
    seg_trainer = WeakSupervisionTrainer()
    seg_trainer.fit(model, train_loader, epochs=20, 
                    lr=1e-4, warm_up=True, cfg=cfg)
    
    print("Re-Trained:", end=' ')
    scores, predictions = seg_trainer.evaluate(model,
                                               datasets['cnn'],
                                               cfg)
    
    df = df.append(scores2df(scores, m='refined', s=s, i=i), ignore_index=True)
    print('\u2713')    
    
    return df



def main():
    
    with open('configs/experiment.config', 'r') as config:
        cfg = eval(config.read())
        
    df = pd.DataFrame(columns=['f1', 'method', 'class', 'set', 'iter'])
    
    # iterate over set 1 and 2 (named differently internally)
    for s in [3]:
        raw_set = AEDataset(cfg, modality='reconstruction', normalize=True,
                            set=s, augment=False, to_gpu=False)
        pca_set = AEDataset(cfg, modality='reconstruction', normalize=False,
                            set=s, augment=False, to_gpu=False)
        cnn_set = AEDataset(cfg, modality='segmentation', normalize=True,
                            set=s, augment=False, to_gpu=True)

        datasets = {'raw': raw_set, 'pca': pca_set, 'cnn': cnn_set}
        
        # 10 runs for each experiment
        for i in range(10):
            tmp = eval_set_for_iter(cfg, s=s, i=i, datasets=datasets)
            df  = df.append(tmp, ignore_index=True)
            df.to_pickle("tmp/tmp_df_set3")
    
    # final clearning for readability
    df['f1']  = df['f1'].apply(lambda x: x.item())
    df['set'] = df['set'] - 1
    df.to_pickle("tmp/table1_data")
    
    
if __name__ == '__main__':
    main()