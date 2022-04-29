from typing import Dict, Iterable, Callable, Generator, Union

from layer import *

class DualBranchAE(nn.Module):
    def __init__(self, encoder, decoder, in_size, n_classes=5, thresholds='learned'):
        super().__init__()
        
        if encoder == 'dual':
            self.encoder = DualLinkEncoder(in_size)

        elif encoder == 'single':
            self.encoder = SingleLinkEncoder(in_size)

        elif encoder == 'zero':
            self.encoder = ZeroLinkEncoder(in_size)

        if decoder == 'reconstruction':
            self.decoder = ReconstructionDecoder()

        elif decoder == 'segmentation':
            self.decoder = SegmentationDecoder(n_classes=n_classes, thresholds=thresholds)
            self.decoder_recon = ReconstructionDecoder()
    
    
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
