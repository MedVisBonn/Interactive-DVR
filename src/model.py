from typing import Dict, Iterable, Callable, Generator, Union

from layer import *



def get_model(
    cfg,
    return_state_dict=False,
    verbose: bool = False
):
    if verbose:
        print(f'Loading model: {cfg.model.encoder} -> {cfg.model.decoder} with spatial dim {cfg.model.spatial_dim}')
    model = DualBranchAE(
        encoder=cfg.model.encoder,
        decoder=cfg.model.decoder,
        in_size=cfg.model.spatial_dim,
    )


    if return_state_dict:
        path = f'{cfg.root_dir}/models/{cfg.model.state_dict}'
        state_dict = torch.load(path)['model_dict']
        if verbose:
            print(f"Done. Returning model and state dict {cfg.model.state_dict}.\n")
        return model, state_dict
    else:
        if verbose:
            print("Done. Returning model only.\n")
        return model



class DualBranchAE(nn.Module):
    def __init__(
        self, 
        encoder, 
        decoder, 
        in_size, 
        n_classes=5, 
        thresholds='learned', 
        recon_channel=288
    ):
        super().__init__()
        
        if encoder == 'dual':
            self.encoder = DualLinkEncoder(in_size)

        elif encoder == 'single':
            self.encoder = SingleLinkEncoder(in_size)

        elif encoder == 'zero':
            self.encoder = ZeroLinkEncoder(in_size)

        if decoder == 'reconstruction':
            self.decoder = ReconstructionDecoder(out_channel=recon_channel)

        elif decoder == 'segmentation':
            self.decoder = SegmentationDecoder(n_classes=n_classes, thresholds=thresholds)
            #self.decoder_recon = ReconstructionDecoder()
    
    
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