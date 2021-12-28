from torch.utils.data import DataLoader
from torch.optim import Adam

import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt


from dataset import *
from model import *
from utils import *
from losses import *
from trainer import SelfSupervisionTrainer


def main():
    # load config
    with open('self_supervision.config', 'r') as config:
        cfg = eval(config.read())
    
    # init logging
    if cfg['log']:
        run = wandb.init(reinit=True, config=cfg, name=cfg['name'], project=cfg['project'])
        cfg = run.config
    
    # init dataset and dataloader
    dataset     = AEDataset(cfg, paper_init=True, modality=cfg['decoder'], 
                            set=cfg['set'], augment=cfg['augment'])
    trainloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    # add some annotations for evaluation
    annot = dataset.initial_annotation()
    dataset.update_annotation(annot)
        
    # init Trainer
    trainer = SelfSupervisionTrainer()
    
    for i in range(3):
        # init Model
        model = DualBranchAE(encoder = cfg['encoder'],
                             decoder = cfg['decoder'],
                             in_size = 145).to(cfg['rank'])

        if cfg['log']:
            wandb.watch(model)
        
        # train model
        trainer.train(model, trainloader, cfg)
        
        # save state dict
        torch.save(model.state_dict(), f'../resources/self-supervision/state_dict-{i}.pt')
        
        
if __name__ == "__main__":
    main()
