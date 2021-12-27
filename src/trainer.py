from torch.utils.data import DataLoader
from torch.optim import Adam

import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt


from dataset import *
from model import *
from utils import *
from losses import *


class SelfSupervisionTrainer(object):
    
    def __init__(self, model, eval_freq=10):
        super().__init__()
        self.model = model
        self.eval_freq = eval_freq
    
    def train(self, trainloader, cfg):
        
        trainloader.dataset.set_mode('train')
        trainloader.dataset.set_modality(cfg['decoder'])
        self.model.train()
        
        if cfg['log']:
            run = wandb.init(reinit=True, config=cfg, name=cfg['name'], project=cfg['project'])
            len_ = trainloader.dataset.__len__()
            bs   = trainloader.batch_size
            mse  = 0.
            
        optimizer   = Adam([{'params': self.model.encoder.parameters(), 'lr': cfg['lr'][0]},
                            {'params': self.model.decoder.parameters(), 'lr': cfg['lr'][1]}])
        
        loss_fn = MSELoss()

        
        for epoch in tqdm(range(cfg['n_epochs'])):
            
            if cfg['augment']:
                trainloader.dataset.update_painting()
                
            for batch in trainloader:
                
                input_ = batch['input']
                target = batch['target']
                mask   = batch['mask']

                output = self.model(input_)
                loss   = loss_fn(output, target, mask.unsqueeze(1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if cfg['log']:
                    mse += loss.cpu().item()
                    
            if cfg['log']:
                mse /= (len_ / bs)
                if epoch % self.eval_freq == 0:
                    scores, _ = self.evaluate(trainloader.dataset, cfg)
                
                wandb.log({'MSE': mse,
                           'RF_scores': scores})
    
    
    def evaluate(self, dataset, cfg):
        
        dataset.augment = False
        layer = 'encoder'
        extractor = FeatureExtractor(self.model, layers=[layer]) 
        features  = extractor(dataset)
        features  = features[layer].permute(0,2,3,1).numpy()
        scores, rf_prediction = evaluate_RF(dataset, features, cfg)
                
        dataset.augment = cfg['augment']
        
        return scores, rf_prediction