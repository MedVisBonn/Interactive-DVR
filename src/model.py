from typing import Dict, Iterable, Callable, Generator, Union

from layer import *

class DualBranchAE(nn.Module):
    def __init__(self, encoder, decoder, in_size, n_classes=5, thresholds='learned', dropout=False, dropout_rate=0.2, recon_channel=288):
        super().__init__()
        
        if encoder == 'dual':
            self.encoder = DualLinkEncoder(in_size, dropout, dropout_rate)

        elif encoder == 'single':
            self.encoder = SingleLinkEncoder(in_size, dropout, dropout_rate)

        elif encoder == 'zero':
            self.encoder = ZeroLinkEncoder(in_size, dropout, dropout_rate)

        if decoder == 'reconstruction':
            self.decoder = ReconstructionDecoder(out_channel=recon_channel, dropout=dropout, dropout_rate=dropout_rate)

        elif decoder == 'segmentation':
            self.decoder = SegmentationDecoder(n_classes=n_classes, thresholds=thresholds)
            #self.decoder_recon = ReconstructionDecoder()
    
    
    def forward_both(self, x):
        assert hasattr(self, 'decoder_recon'), 'reconstruction decoder not found'
        x_encoded = self.encoder(x)
        x_segment = self.decoder(x_encoded)
        x_recon   = self.decoder_recon(x_encoded)
        
        return x_segment, x_recon
    
    
    def forward_features(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        
        return x_decoded, x_encoded
    
    
    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        
        return x_decoded
    
    
    @torch.no_grad()
    def zero_cross_connections(self):
        for name, params in self.named_parameters():
            if 'pre' in name:
                params.requires_grad_(False)
                params.zero_()
                
                
    @torch.no_grad()
    def activate_cross_connections(self):
        for name, params in self.named_modules():
            if 'pre' in name:
                params.requires_grad_(True)
                params.reset_parameters()
                
                
    @torch.no_grad()           
    def load_encoder_state(self, state_dict: Dict) -> None:
        # load all params that don't cause problems
        incompatible, unexpected = self.encoder.load_state_dict(state_dict, strict=False)
        # clean incompatible and unexpected keys. Ignore cross connections and
        # remove keys related to batch norms
        incompatible = [s for s in incompatible if 'pre' not in s]
        unexpected   = [s for s in unexpected   if 'num_batches' not in s]
        # extract state dict for changes
        original_state_dict = self.encoder.state_dict()
        # add missing params to original state dict
        for target_param, source_param in zip(incompatible, unexpected):
            original_state_dict[target_param] = state_dict[source_param]
        # load state dict back into model
        self.encoder.load_state_dict(original_state_dict)