from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt


from dataset import *
from model import *
from utils import *
from losses import *


class SelfSupervisionTrainer(object):
    
    def __init__(self):
        super().__init__()
    
    def train(self, model, trainloader, cfg):
        
        trainloader.dataset.set_mode('train')
        trainloader.dataset.set_modality(cfg['decoder'])
        model.train()
        
        if cfg['log']:

            len_ = trainloader.dataset.__len__()
            bs   = trainloader.batch_size
            mse  = 0.
            
        optimizer   = Adam([{'params': model.encoder.parameters(), 'lr': cfg['lr'][0]},
                            {'params': model.decoder.parameters(), 'lr': cfg['lr'][1]}])
        
        loss_fn = MSELoss()

        
        for epoch in tqdm(range(1, cfg['n_epochs']+1)):

            if cfg['augment']:
                trainloader.dataset.update_painting()

            if epoch == 50:
                for g in optimizer.param_groups:
                    g['lr'] = 0.00001
                
            for batch in trainloader:
                
                input_ = batch['input']
                target = batch['target']
                mask   = batch['mask']

                output = model(input_)
                loss   = loss_fn(output, target, mask.unsqueeze(1))
                
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
            
                if cfg['log']:
                    mse += loss.cpu().item()
                        
            if cfg['log']:
                mse /= (len_ / bs)
                if epoch % cfg['eval_freq'] == 0:
                    scores, _ = self.evaluate(model, trainloader.dataset, cfg)
                
                wandb.log({'MSE': mse,
                           'RF_scores': scores})
    
    
    def evaluate(self, model, dataset, cfg):
        
        dataset.augment = False
        layer = 'encoder'
        extractor = FeatureExtractor(model, layers=[layer]) 
        features  = extractor(dataset)
        features  = features[layer].permute(0,2,3,1).numpy()
        scores, rf_prediction = evaluate_RF(dataset, features, cfg)
                
        dataset.augment = cfg['augment']
        
        return scores, rf_prediction