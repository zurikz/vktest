from torch import nn
from torch import Tensor
from typing import Tuple

from .modules import InstanceNorm2d, AgainConv1d


class DecoderConvBlock(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
	) -> None:
		super().__init__()
		self.convblock, self.genblock = 2 * [
			nn.Sequential(
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
		]
	
	def forward(self, x: Tensor) -> Tensor:
		y = self.convblock(x)
		y += self.genblock(y)
		return x + y

class Decoder(nn.Module):
	"""AGAIN-VC decoder block.

	Args:
		melbins (int): number of input channels.
			In our terms it's the number of Mel bins.
		melbins_out (int): number of output channels.
		hidden_size (int): hidden size of channels.
		conv_blocks_num (int): number of ConvBlocks in decoder.
	"""
	def __init__(
		self,
		melbins: int,
		melbins_out: int,
		hidden_size: int,
		conv_blocks_num: int
	) -> None:
		super().__init__()
		self.first_conv1d = AgainConv1d(
			in_channels=melbins, out_channels=hidden_size
		)
		self.leakyrelu = nn.LeakyReLU()

		self.convblocks = nn.ModuleList(
			conv_blocks_num * [
				DecoderConvBlock(in_channels=hidden_size, out_channels=hidden_size)
			]
		)

		self.IN = InstanceNorm2d()
		self.gru = nn.GRU(
			input_size=hidden_size,
			hidden_size=hidden_size,
			num_layers=2
		)
		self.linear = nn.Linear(hidden_size, melbins_out)
	
	def forward(
		self,
		source: Tuple[Tensor, Tensor, Tensor],
		target: Tuple[Tensor, Tensor, Tensor]
	) -> Tensor:
		"""Decoder's forward pass.

		Args:
			source (tuple): (content, means, stds)
			target (tuple): (content, means, stds)
				content (Tensor): (B, melbins, seglen)
				means, stds (Tensor): (B, conv_blocks_num)
		
		Returns:
		"""
		source_content, _, _ = source
		target_content, means, stds = target

		# Adaptive Instance Normalization
		source_content, _, _ = self.IN(source_content)
		_, target_content_mean, target_content_std = self.IN(target_content)
		stylized_content = target_content_std * source_content + target_content_mean

		# (B, melbins, seglen) -> (B, hidden_size, seglen)
		stylized_content = self.first_conv1d(stylized_content)
		output = self.leakyrelu(stylized_content)

		# Flip tensors with means (and stds) to use
		# means from the last layers of encoder first.
		means, stds = means.flip(dims=[1]), stds.flip(dims=[1])
		for (block, mean, std) in zip(self.convblocks, means, stds):
			output = block(output)
			# Adaptive Instance Normalization
			output, _, _ = self.IN(output)
			output = std * output + mean
		
