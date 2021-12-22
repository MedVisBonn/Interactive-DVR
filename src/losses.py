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