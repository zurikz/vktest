import torch
from torch import nn
from torch import Tensor
from typing import Tuple

from .modules import InstanceNorm, AgainConv1d


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
		y = y.clone() + self.genblock(y)
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
				DecoderConvBlock(
					in_channels=hidden_size, 
					out_channels=hidden_size
				)
			]
		)

		self.IN = InstanceNorm()
		self.gru = nn.GRU(
			input_size=hidden_size,
			hidden_size=hidden_size,
			num_layers=2
		)
		self.linear = nn.Linear(hidden_size, melbins_out)
	
	def forward(
		self,
		source_content: Tensor,
		target: Tuple[Tensor, Tensor, Tensor]
	) -> Tensor:
		"""Decoder's forward pass.

		Args:
			source_content (tuple): source content representation.
				Shape: (B, melbins, seglen)
			target (tuple): (content, means, stds)
				content (Tensor): (B, melbins, seglen)
				means, stds (Tensor): (conv_blocks_num, B, melbins, 1)
		
		Returns:
		"""
		target_content, means, stds = target

		# Adaptive Instance Normalization
		source_content, _, _ = self.IN(source_content)
		_, target_mean, target_std = self.IN(target_content)
		stylized_content = target_std * source_content + target_mean

		# (B, melbins, seglen) -> (B, hidden_size, seglen)
		stylized_content = self.first_conv1d(stylized_content)
		output = self.leakyrelu(stylized_content)

		# Flip tensors of means (and stds) along "conv_blocks_num"
		# to use means from the last layers of encoder first.
		means, stds = means.flip(dims=[0]), stds.flip(dims=[0])
		for (block, mean, std) in zip(self.convblocks, means, stds):
			output = block(output)
			# Adaptive Instance Normalization
			output, _, _ = self.IN(output)
			output = std * output + mean

		# output: (B, hidden_size, seglen)
		output = torch.cat([means[-1], output], dim=2)
		# (B, hidden_size, seglen) -> (B, seglen, hidden_size)
		output = output.transpose(dim0=1, dim1=2)
		output, _ = self.gru(output)
		output = output[:, 1:, :]
		output = self.linear(output)
		return output.transpose(dim0=1, dim1=2)

