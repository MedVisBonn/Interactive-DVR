import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def forward(self, output, target, weight):
        """
        Input: model output (decoded), targets, sparsity weight matrix
        Output: loss
        """
        loss_pre = F.mse_loss(output, target, reduction="none")
        loss     = (loss_pre * weight).sum() / max(1, weight.sum())
        
        return loss
    

class SEGLoss(nn.Module):
    
    def __init__(self, focal=True, gamma=2.):
        super().__init__()
        self.focal = focal
        self.gamma = gamma
    
    def forward(self, input_, target, weight, pos_weight, scaler):
        loss  = F.binary_cross_entropy_with_logits(
                    input      = input_,
                    target     = target,
                    weight     = weight,
                    pos_weight = pos_weight,
                    reduction  = "sum")
        
        if self.focal:
            p_t  = torch.exp(-loss)
            loss = (1-p_t) ** self.gamma * loss
            
        return loss / scaler


class ThresholdRegularizer(nn.Module):
    
    def __init__(self, gamma=1):
        super().__init__()
        self.gamma = gamma
    
    def forward(self, x):
        return self.gamma * torch.sum( 0.25 - (x - 0.5) ** 2 )
    
    
class FeatureRegularizer(nn.Module):

    def __init__(self, mode='entropy', alpha=1e-5):
        super().__init__()
        self.mode = mode
        self.alpha = alpha

    def forward(self, feature, mask):
        f = feature.permute(0,2,3,1).view(-1, 44)[mask.view(-1) == 1]
        #f = feature
        if f.numel():
            f = torch.tanh(f)
            f = (f + 1.) / 2.
            f = F.normalize(f, p=1)
        
            if self.mode == 'entropy':
                #avg_f = f.mean(0)
                return self.alpha * (- f * torch.log(f + 1e-4)).mean(1).mean()

            elif self.mode == 'hoyer':
                return (torch.norm(f, dim=0, p=1) / torch.norm(f, dim=0, p=2)).mean()

        else:
            return torch.tensor(0., requires_grad=True)