from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, Adam, lr_scheduler
from torch.nn.utils import clip_grad_norm_

import wandb
from tqdm import tqdm

from dataset import *
from model import *
from utils import *
from losses import *



class Trainer:
        
    @torch.no_grad()
    def evaluate(self, model: nn.Module, dataset: Dataset, cfg: Dict[str, object]) \
        -> Union[Dict[str, Tensor], Tensor]:
    
        augment_checkpoint = dataset.augment
        dataset.augment = False
        layer = 'encoder'
        extractor = FeatureExtractor(model, layers=[layer]) 
        features  = extractor(dataset)
        features  = features[layer].permute(0,2,3,1).numpy()
        scores, rf_prediction = evaluate_RF(dataset, features, cfg)
                
        dataset.augment = augment_checkpoint
        
        return scores, rf_prediction
    
    
    @torch.no_grad()
    def model_dice(self, model: nn.Module, dataset: Dataset, mode: str):
        
        assert dataset.augment == False, "Turn off augmentation when evaluating"
        assert dataset.modality == 'segmentation', "Dataset not in segmentation mode"
        assert model.decoder.__class__.__name__ == 'SegmentationDecoder', \
        "Model doesn't have a segmentation head"
        
        mode_checkpoint = dataset.mode
        dataset.set_mode(mode)
        model.eval()
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        pos_weight = dataloader.dataset.pos_weight


        eps = 1e-5    
        predictions = []

        for batch in dataloader:
            input_ = batch['input']
            target = batch['target']
            mask   = batch['weight']
            output = model(input_)

            predictions.append((output.detach()>0).detach().cpu())

        if mode == 'train':
            target = dataloader.dataset.annotations.detach().cpu()
        else:
            target = dataloader.dataset.label.detach().cpu()

        mask       = ((target.sum(0, keepdim=True) > 0) * 1)
        prediction = torch.cat(predictions, dim=0).permute(1,0,2,3)#.detach()


        TP = (prediction * target * mask).sum((1,2,3))
        TPplusFP = (prediction*mask).sum((1,2,3))
        TPplusFN = (target*mask).sum((1,2,3))
        set_dice = (2*TP + eps) / (TPplusFP + TPplusFN + eps)
        
        dataset.set_mode(mode_checkpoint)

        return set_dice, prediction


class SelfSupervisionTrainer(Trainer):
    
    def __init__(self):
        super().__init__()
    
    def train(self, model: nn.Module, trainloader: DataLoader, epochs: int,
              cfg: Dict[str, object]) -> None:
        
        trainloader.dataset.set_mode('train')
        trainloader.dataset.set_modality('reconstruction')

        hook = OutputHook()

        model.train()
        
        if cfg['log']:
            len_ = trainloader.dataset.__len__()
            bs   = trainloader.batch_size
            mse  = 0.
            reg  = 0.
            
        optimizer   = Adam([{'params': model.encoder.parameters(), 'lr': cfg['s_lr'][0]},
                            {'params': model.decoder.parameters(), 'lr': cfg['s_lr'][1]}])
        
        loss_fn = MSELoss()
        regularizer = FeatureRegularizer(alpha=10.)
        
        for epoch in tqdm(range(1, epochs+1)):
            
            handle = model.encoder.register_forward_hook(hook)

            if cfg['log']:
                mse = 0.
                reg = 0.
                
            if cfg['augment']:
                trainloader.dataset.update_painting()

            if epoch == 50:
                for g in optimizer.param_groups:
                    g['lr'] = 0.00001
                
            for batch in trainloader:
                
                input_ = batch['input']
                target = batch['target']
                mask   = batch['mask']
                
                output = model.forward(input_)
                MSE    = loss_fn(output, target, mask.unsqueeze(1))
                #REG    = regularizer(hook.output, mask)
                loss   = MSE # + REG    
                
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 2)
                optimizer.step()
                
                if cfg['log']:
                    mse += MSE.cpu().item()
                    #reg += REG.cpu().item()
            
            handle.remove()
            
            if cfg['log']:
                mse /= (len_ / bs)
                reg /= (len_ / bs)
                if epoch % cfg['s_eval_freq'] == 0:
                    scores, _ = self.evaluate(model, trainloader.dataset, cfg)
                    wandb.log({'RF-training': scores}, commit=False)

                wandb.log({'MSE-training': mse,
                           'REG-training': reg})


    
    
class WeakSupervisionTrainer(Trainer):
    
    def __init__(self):
        super().__init__()
        
    def train(self, model: nn.Module, trainloader: DataLoader, epochs: int,
              lr: list, warm_up: bool, cfg: Dict[str, object]) -> None:
        
        trainloader.dataset.augment = False
        mode_checkpoint = trainloader.dataset.mode
        trainloader.dataset.set_mode('train')
        trainloader.dataset.set_modality('segmentation')
        
        hook = OutputHook()
                
        if cfg['log']:

            len_ = trainloader.dataset.__len__()
            bs   = trainloader.batch_size
            
        #optimizer   = Adam([{'params': model.encoder.parameters(), 'lr': lr[0]},
        #                    {'params': model.decoder.parameters(), 'lr': lr[1]},
        #                    {'params': model.decoder_recon.parameters(), 'lr': lr[1]}])
        
        optimizer   = Adam(model.parameters(), lr=lr)
        #scheduler   = lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr/100, verbose=True)
        #scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda epoch: 0.9**epoch, verbose=True)
        loss_fn     = SEGLoss()
        recon_loss  = MSELoss()
        regularizer = ThresholdRegularizer(gamma=1e-6)
        
        scaler      = trainloader.dataset.annotations.sum().to(cfg['rank'])
        pos_weight  = trainloader.dataset.pos_weight.to(cfg['rank'])
        
        for epoch in tqdm(range(0, epochs)):
            

            # register hook for extracting thresholded features
            handle = model.decoder.threshold.register_forward_hook(hook)
            
            # Gradient Freezing
            model.train()
            
            if warm_up:
                if epoch <= 5:       
                    for param in model.encoder.parameters():
                        param.requires_grad = False
                    model.encoder.eval()

                if epoch == 6:
                    for param in model.parameters():
                        param.requires_grad = True
            
            else:
                for param in model.parameters():
                    param.requires_grad = True
                    
                    
            if (warm_up and epoch == 15) or (not warm_up and epoch == 10):
                for g in optimizer.param_groups:
                    g['lr'] = 0.00001
                    
                    
            if (not warm_up and epoch == 30):
                for g in optimizer.param_groups:
                    g['lr'] = 0.000001
                    
            # training loop
            for batch in trainloader:

                input_ = batch['input']
                target = batch['target']
                weight = batch['weight']
                mask   = batch['mask']
                
                output, recon = model.forward_both(input_)
                loss   = loss_fn(output, target, weight, 
                                 pos_weight, scaler) \
                         + regularizer(hook.output) \
                         + 0.1*recon_loss(recon, input_.detach(), mask.unsqueeze(1))

                optimizer.zero_grad()          
                loss.backward()
                optimizer.step()
                
            #if (not warm_up) or (epoch > 5 and warm_up):
            #    scheduler.step()
            
            #remove hook at the end of training due to interaction with evaluation hooks
            handle.remove()
            
            if cfg['log']:
                model.eval()
                if epoch % cfg['w_eval_freq'] == 0:
                    scores, _ = self.evaluate(model, trainloader.dataset, cfg)
                    model_train_scores, _ = self.model_dice(model, trainloader.dataset, 'train')
                    model_test_scores, _  = self.model_dice(model, trainloader.dataset, 'validate')
                wandb.log({'RF-training': scores,
                           'Model-training-Dice': model_train_scores.detach().cpu()[1:].mean(),
                           'Model-validation-Dice': model_test_scores.detach().cpu()[1:].mean()})
                
        trainloader.dataset.set_mode(mode_checkpoint)