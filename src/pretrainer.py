import os, sys
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from typing import Dict, Iterable, Callable, Generator, Union


from utils import *


class PreTrainer():
    def __init__(self, model, criterion, train_loader, cfg,
                valid_loader=None, eval_metrics=None, lr=5e-4, patience=5, es_mode='min', 
                description='untitled', n_epochs=10000, log=False):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.description = description
        self.cfg = cfg
        self.n_epochs = n_epochs
        
        self.model  = model.to(self.device)

        self.criterion = criterion
        self.train_loader = train_loader
        self.train_loader.dataset.set_modality('reconstruction')
        self.valid_loader = train_loader if valid_loader is None else valid_loader
        self.valid_loader.dataset.set_modality('reconstruction')
        self.lr = lr
        self.patience = patience
        self.eval_metrics = eval_metrics
        self.es_mode = es_mode
        #self.optimizer = torch.optim.Adam([param for name, param in model.named_parameters() if not all([s in name for s in ['layer', 'pre_conv']])], lr=lr)
        self.optimizer = torch.optim.Adam([
            {'params': self.model.encoder.parameters(), 'lr': lr},
            {'params': self.model.decoder.parameters(), 'lr': lr, 'weight_decay': 0}])

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=self.patience)
        self.es = EarlyStopping(mode=self.es_mode, patience=2*self.patience)
        self.history = {'train loss': [], 'valid loss' : [], 'rf scores': []}
        if self.eval_metrics is not None:
            self.history = {**self.history, **{key: [] for key in self.eval_metrics.keys()}}
        self.training_time = 0
        
        self.log = log
        if self.log:
            run = wandb.init(reinit=True, name='log_' + self.description, project='PreTrainer_test')
            #wandb.watch(self.model, log='all', log_freq=1)
        
    def inference_step(self, x):
        return self.model(x)
    
    def save_hist(self):
        if(not os.path.exists("trainer_logs")):
            os.makedirs("trainer_logs")
        savepath = f"trainer_logs/{self.description}.npy"
        np.save(savepath, self.history)
        return
    
    def save_model(self):
        if(not os.path.exists("models")):
            os.makedirs("models")
        if(not os.path.exists("trainer_logs")):
            os.makedirs("trainer_logs")
        savepath = f"models/{self.description}_best.pt"
        torch.save({
        'model_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        }, savepath)
        self.save_hist()
        return
    
    def load_model(self):
        savepath = f"models/{self.description}_best.pt"
        checkpoint = torch.load(savepath)
        self.model.load_state_dict(checkpoint['model_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        savepath = f"trainer_logs/{self.description}.npy"
        self.history = np.load(savepath,allow_pickle='TRUE').item()
        return
    
    def train_epoch(self):
        loss_list, batch_sizes = [], []
        for batch in self.train_loader:
            input_ = batch['input']#.to(self.device)
            target = batch['target']#.to(self.device)
            mask   = batch['mask']#.to(self.device)
            net_out = self.inference_step(input_)
            
            loss = self.criterion(net_out, target, mask.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            loss_list.append(loss.item())
            batch_sizes.append(target.shape[0])
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['train loss'].append(average_loss)
        
        if self.log:
            wandb.log({
                'train_loss': average_loss
            }, commit=False)
        
        
        return average_loss
    
    @torch.no_grad()
    def evaluate_rf(self) -> Union[Dict[str, Tensor], Tensor]:
        
        augment_checkpoint = self.train_loader.dataset.augment
        self.train_loader.dataset.augment = False
        layer = 'encoder'
        extractor = FeatureExtractor(self.model, layers=[layer]) 
        features  = extractor(self.train_loader.dataset)
        features  = features[layer].permute(0,2,3,1).numpy()
        scores, rf_prediction = evaluate_RF(self.train_loader.dataset, features, self.cfg)
                
        self.train_loader.dataset.augment = augment_checkpoint
        
        return scores, rf_prediction
    
    
    @torch.no_grad()
    def eval_epoch(self):
        loss_list, batch_sizes = [], []
        if self.eval_metrics is not None:
            epoch_metrics = {key: [] for key in self.eval_metrics.keys()}

        for batch in self.valid_loader:
            input_ = batch['input']#.to(self.device)
            target = batch['target']#.to(self.device)
            mask   = batch['mask']#.to(self.device)
            net_out = self.inference_step(input_)
            loss = self.criterion(net_out, target, mask.unsqueeze(1))
            
            loss_list.append(loss.item())
            batch_sizes.append(target.shape[0])
            if self.eval_metrics is not None:
                for key, metric in self.eval_metrics.items():
                    epoch_metrics[key].append(metric(torch.atleast_2d(net_out),target,weight).item())
        average_loss = epoch_average(loss_list, batch_sizes)
        self.history['valid loss'].append(average_loss)
        if self.eval_metrics is not None:
            for key, epoch_scores in epoch_metrics.items():
                self.history[key].append(epoch_average(epoch_scores, batch_sizes))
        return average_loss
    
    
    def fit(self):
        best_es_metric = 1e25 if self.es_mode == 'min' else -1e25
        self.model.eval()
        valid_loss = self.eval_epoch()
        progress_bar = tqdm(range(self.n_epochs), total=self.n_epochs, position=0, leave=True)
        self.training_time = time()
        for epoch in progress_bar:      
            self.model.train()
            train_loss = self.train_epoch()
            
            self.model.eval()
            valid_loss = self.eval_epoch()
            self.scheduler.step(valid_loss)
            scores, rf_prediction = self.evaluate_rf()
            self.history['rf scores'].append(scores['Avg_f1_tracts'])
            if self.log:
                wandb.log({'metrics': scores})
            
            epoch_summary = [f"Epoch {epoch+1}"] + [f" - {key}: {self.history[key][-1]:.4f} |" for key in self.history] + [ f"ES epochs: {self.es.num_bad_epochs}"]
            progress_bar.set_description("".join(epoch_summary))
            es_metric = list(self.history.values())[1][-1]
            if self.es_mode == 'min':
                if es_metric < best_es_metric:
                    best_es_metric = es_metric
                    self.save_model()
            else:
                if es_metric > best_es_metric:
                    best_es_metric = es_metric
                    self.save_model()
            if(self.es.step(es_metric)):
                print('Early stopping triggered!')
                break
                
        self.training_time = time() - self.training_time
        self.save_hist()
        self.load_model()
        