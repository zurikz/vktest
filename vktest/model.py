import torch
from torch import nn
from torch import Tensor
import torch.functional as F

from .encoder import Encoder
from .decoder import Decoder
from .modules import CustomSigmoid


class AgainVC(nn.Module):
    """AgainVC model.

    Args:
        encoder_params (dict): dict with encoder params.
        decoder_params (dict): dict with decoder params.
    """
    def __init__(
        self,
        encoder_params: dict,
        decoder_params: dict
    ) -> None:
        super().__init__()
        self.encoder = Encoder(**encoder_params)
        self.decoder = Decoder(**decoder_params)
        self.sigmoid = CustomSigmoid()

    def forward(self, source: Tensor, target: Tensor = None) -> Tensor:
        """AgainVC's forward pass.

        Args:
            source (Tensor): source spectrogram to extract content.
                Shape: (B, melbins, seglen)
            target (Tensor): target spectrogram to extract style.
                Shape: (B, melbins, seglen)

        Returns:

        """
        if target is None:
            seglen = source.shape[2]
            target = torch.cat(
                (source[:, :, (seglen // 2):], source[:, :, :(seglen // 2)]),
                axis=2
            )
        
        source_content, _, _ = self.encoder(source)
        target_content, target_means, target_stds = self.encoder(target)

        source_content = self.sigmoid(source_content)
        target_encoded = (self.sigmoid(target_content), target_means, target_stds)

        return self.decoder(source_content, target_encoded)

    def inference(self, source: Tensor, target: Tensor) -> Tensor:
        source_len, target_len = source.shape[2], target.shape[2]

        if source_len % 8 != 0:
            source = F.pad(source, (0, 8 - source_len % 8), mode='reflect')
        if target_len % 8 != 0:
            target = F.pad(target, (0, 8 - target_len % 8), mode='reflect')
        
        result = self.forward(source, target)

        return result[:, :, :source_len]
