import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt


from dataset import *
from model import *
from utils import *
from losses import *
from trainer import *


def run(i):
    with open('configs/experiment.config', 'r') as config:
        cfg = eval(config.read())

    if cfg['log']:
        run = wandb.init(reinit=True, config=cfg, name=cfg['name'], project=cfg['project'])
        cfg = run.config


    dataset = AEDataset(cfg, modality='reconstruction', 
                        set=cfg['set'], augment=cfg['augment'])

    dataset.clear_annotation()
    data_path = '../experiments/2021-12-27/reconstruction/data/Tensor-0-1000-300.pt'
    annot     = torch.load(data_path)
    dataset.update_annotation(annot)

    
    model_dir = '../trained-models/reconstruction/'
    #dict_name = f"{cfg['name']}-{cfg['s_encoder']}-{cfg['augment']*'augment-'}{i}.pt"
    dict_name = f"App-Test-{cfg['s_encoder']}-{cfg['augment']*'augment-'}{i}.pt"
    
    dict_path = model_dir + dict_name

    db_recon = DualBranchAE(encoder = cfg['s_encoder'],
                            decoder = 'reconstruction',
                            in_size = 145).to(cfg['rank'])
    
    trainer_recon = SelfSupervisionTrainer()

    if cfg['s_load']: #os.path.isfile(dict_path) and 
        db_recon.load_state_dict(torch.load(dict_path), strict=True)
        #db_recon.load_state_dict(torch.load(f'../resources/self-supervision/state_dict-{i}.pt'))

    else:
        trainloader = DataLoader(dataset, batch_size=cfg['s_batch_size'],
                                 shuffle=True)
        trainer_recon.train(db_recon, trainloader, cfg['s_n_epochs'], cfg)
        torch.save(db_recon.state_dict(), dict_path)


    data_dir  = '../data/initial-interaction/'
    data_name = f"{cfg['name']}-{cfg['s_encoder']}-{cfg['augment']*'augment-'}"\
                f"{cfg['num_interactions']}-{i}.pt"
    data_path = data_dir + data_name

    dataset.clear_annotation()

    if os.path.isfile(data_path) and cfg['data_load']:
        annot = torch.load(data_path)
        dataset.update_annotation(annot)

    else:
        annot = dataset.initial_annotation()
        dataset.update_annotation(annot)

        for j in range(cfg['num_interactions']):
            scores, predictions = trainer_recon.evaluate(db_recon,
                                                         dataset,
                                                         cfg)
            if cfg['log']:
                wandb.log({'RF_s'  : scores,
                           'x': j})     

            annot = dataset.refinement_annotation(predictions)
            dataset.update_annotation(annot)

        torch.save(dataset.annotations.cpu().detach(), data_path)    

        scores_checkpoint = scores.copy()


    annot_checkpoint = dataset.annotations.cpu().detach().clone()

    for j in range(cfg['num_interactions']):
        scores, predictions = trainer_recon.evaluate(db_recon,
                                                     dataset,
                                                     cfg)
        if cfg['log']:
            wandb.log({'RF_s'  : scores,
                       'x': j+cfg['num_interactions']})

        annot = dataset.refinement_annotation(predictions)
        dataset.update_annotation(annot)

    scores, predictions = trainer_recon.evaluate(db_recon,
                                                 dataset,
                                                 cfg)
    if cfg['log']:
        wandb.log({'RF_s'  : scores,
                   'x': 2*cfg['num_interactions']})

    dataset.clear_annotation()
    dataset.update_annotation(annot_checkpoint)


    model_dir = '../trained-models/segmentation/'
    dict_name = f"{cfg['name']}-{cfg['w_encoder']}-{i}.pt"
    dict_path = model_dir + dict_name

    db_segment = DualBranchAE(encoder    = cfg['w_encoder'],
                              decoder    = 'segmentation',
                              in_size    = 145,
                              n_classes  = len(cfg['labels']),
                              thresholds = cfg['thresholds']).to(cfg['rank'])

    trainer_segment = WeakSupervisionTrainer()
    trainloader = DataLoader(dataset, batch_size=cfg['w_batch_size'], 
                             shuffle=True)
    
    if os.path.isfile(dict_path) and cfg['w_load']:
        db_segment.load_state_dict(torch.load(dict_path), strict=True)

    else:
        db_segment.encoder.load_state_dict(db_recon.encoder.state_dict())
        dataset.augment  = False
        dataset.modality = 'segmentation'

        trainer_segment.train(db_segment, trainloader, epochs=cfg['w_n_epochs'], 
                              lr=cfg['w_lr'], warm_up=True, cfg=cfg)
        torch.save(db_segment.state_dict(), dict_path)



    annot_checkpoint = dataset.annotations.cpu().detach().clone()

    if cfg['log'] and not cfg['data_load']:
        wandb.log({'RF_w'  : scores_checkpoint,
                   'x': cfg['num_interactions']-1})

    for j in range(cfg['num_interactions']):
        scores, predictions = trainer_segment.evaluate(db_segment,
                                                       dataset,
                                                       cfg)
        
        #trainer_segment.train(db_segment, trainloader, epochs=9, 
        #                      lr=0.0001, warm_up=False, cfg=cfg)

        if cfg['log']:
            wandb.log({'RF_w'  : scores,
                       'x': j+cfg['num_interactions']})

        annot = dataset.refinement_annotation(predictions)
        dataset.update_annotation(annot)
        
        trainer_segment.train(db_segment, trainloader, epochs=10, 
                              lr=0.0001, warm_up=False, cfg=cfg)

    scores, predictions = trainer_segment.evaluate(db_segment,
                                                   dataset,
                                                   cfg)
    if cfg['log']:
        wandb.log({'RF_w'  : scores,
                   'x': 2*cfg['num_interactions']})

    dataset.clear_annotation()
    dataset.update_annotation(annot_checkpoint)
    
    
def main():
    for i in range(10):
        run(i)
    
if __name__ == '__main__':
    main()