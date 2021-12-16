from typing import Dict, Iterable, Callable, Generator, Union

from layer import *

class DualBranchAE(nn.Module):
    def __init__(self, encoder, decoder, in_size, n_classes=None, thresholds='learned'):
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
            self.decoder = SegmentationDecoder(num_classes=5, thresholds=thresholds)


    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)

        return x_decoded
