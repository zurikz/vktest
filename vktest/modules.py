from torch import nn
from torch import Tensor
from typing import Tuple


class InstanceNorm2d(nn.Module):
	"""Instance normalization layer.
	"""
	def __init__(self) -> None:
		super().__init__()
	
	def forward(self, x) -> Tuple[Tensor, Tensor, Tensor]:
		"""
		Args:
			x (Tensor): (batch, channel, height, width)

		Returns:
			tuple: (IN(x), mean, std)
		"""
		batch, channel = x.shape[0], x.shape[1]
		mean = x.view(batch, channel, -1).mean(-1)
		std = (x.view(batch, channel, -1).var(-1) + 1e-5).sqrt()
		mean = mean.view(batch, channel, 1, 1)
		std = std.view(batch, channel, 1, 1)
		x = (x - mean) / std
		return x, mean, std

class AgainConv1d(nn.Module):
	"""Conv1d layer with Xavier uniform initialization.
	"""
	def __init__(self, in_channels: int, out_channels: int,
				 kernel_size: int = 1, stride: int = 1, 
				 padding: int = None, dilation: int = 1,
				 groups: int = 1, bias: bool = True) -> None:
		super().__init__()
		if padding is None:
			assert(kernel_size % 2 == 1)
			# Set padding to preserve equality of L_in and L_out,
			# where L is a length of signal sequence.
			padding = int(dilation * (kernel_size - 1) / 2)
		
		self.conv1d = nn.Conv1d(
			in_channels=in_channels, out_channels=out_channels,
			kernel_size=kernel_size, stride=stride,
			padding=padding, dilation=dilation,
			groups=groups, bias=bias, padding_mode='zeros'
		)

		nn.init.xavier_uniform_(
			self.conv1d.weight,
			gain=nn.init.calculate_gain('linear')
		)
	
	def forward(self, x) -> Tensor:
		return self.conv1d(x)
