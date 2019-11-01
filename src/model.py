import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Encoder_block, Decoder_block

example_architecture = {
	"kernel_size":  3,
	"padding": 1, 
	"stride": 1,
	"pool_dims": (2,2), 
	"encoder_blocks":
	[
		(3, 17, 2), #<- Data structure: (input channels, output channels, number of layers) as you please, Master
		(29, 37, 2),
		(37, 37, 3),
	],
	"decoder_blocks":
	[	
		(37, 37, 2),
		(37, 29, 2),
		(17, 3, 3)
	],
}


class Net(nn.Module):
	def __init__(self, architecture_dict):
		super().__init__()

		self.kernel_size = architecture_dict["kernel_size"]
		self.padding = architecture_dict["padding"]
		self.stride = architecture_dict["stride"]
		self.pool_dims = architecture_dict["pool_dims"]

		
		encoder_layers = []
		decoder_layers = []

		for encoder_sizes in architecture_dict["encoder_blocks"]:
			encoder_layers.append(
				Encoder_block(*encoder_sizes, self.kernel_size, self.padding, self.stride, self.pool_dims)
			)
			
		for decoder_sizes in architecture_dict["decoder_blocks"]:
			decoder_layers.append(
				Decoder_block(*decoder_sizes, self.kernel_size, self.padding, self.stride, self.pool_dims)
			)

		self.encoder = nn.Sequential(*encoder_layers)
		self.decoder = nn.Sequential(*decoder_layers)


	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return F.softmax(x)




#net = Net(example_architecture)
#print(net)