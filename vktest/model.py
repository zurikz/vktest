import torch
from torch import nn
from torch import Tensor
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
        seglen = source.shape[2]

        if target is None:
            target = torch.cat(
                (source[:, :, (seglen // 2):], source[:, :, :(seglen // 2)]),
                axis=2
            )
        
        source_content, _, _ = self.encoder(source)
        target_content, target_means, target_stds = self.encoder(target)

        source_content = self.sigmoid(source_content)
        target_encoded = (self.sigmoid(target_content), target_means, target_stds)

        return self.decoder(source_content, target_encoded)
