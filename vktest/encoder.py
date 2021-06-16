from torch import nn
from torch import Tensor
from typing import Tuple

from .modules import InstanceNorm2d, AgainConv1d 


class EncoderConvBlock(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
	) -> None:
		super().__init__()
		self.convblock = nn.Sequential(
			AgainConv1d(
				in_channels=in_channels,
				out_channels=out_channels,
				kernel_size=3
			),
			nn.BatchNorm1d(out_channels),
			nn.LeakyReLU(),
			AgainConv1d(
				in_channels=out_channels,
				out_channels=in_channels,
				kernel_size=3
			)
		)

	def forward(self, x: Tensor) -> Tensor:
		y = self.convblock(x)
		return x + y

class Encoder(nn.Module):
	"""AGAIN-VC encoder block.

	Args:
		melbins (int): number of input channels.
			In our terms it's the number of Mel bins.
		melbins_out (int): number of output channels.
		hidden_size (int): hidden size of channels.
		conv_blocks_num (int): number of ConvBlocks in encoder.
	"""
	def __init__(
		self,
		melbins: int,
		melbins_out: int,
		hidden_size: int,
		conv_blocks_num: int
	) -> None:
		super().__init__()
		self.IN = InstanceNorm2d()
		self.first_conv1d = AgainConv1d(
			in_channels=melbins, out_channels=hidden_size
		)
		self.convblocks = nn.ModuleList(
			conv_blocks_num * [
				EncoderConvBlock(in_channels=hidden_size, out_channels=hidden_size)
			]
		)
		self.last_conv1d = AgainConv1d(
			in_channels=hidden_size, out_channels=melbins_out
		)
	
	def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
		"""Encoder's forward pass.

		Args:
			x (Tensor): (batch, melbins, seglen)
		
		Returns:
			tuple:	(encoder_out, means, stds)
				encoder_out: (batch, melbins_out, seglen)
		"""
		# (B, melbins, seglen) -> (B, hidden_size, seglen)
		x = self.first_conv1d(x)

		means, stds = [], []
		for block in self.convblocks:
			x = block(x)
			x, mean, std = self.IN(x)
			means.append(mean)
			stds.append(std)
		
		# (B, hidden_size, seglen) -> (B, melbins_out, seglen)
		x = self.last_conv1d(x)

		return x, means, stds
