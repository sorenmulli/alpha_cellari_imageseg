import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import EncoderBlock, DecoderBlock

example_architecture = {
	"kernel_size":  3,
	"padding": 1, 
	"stride": 1,
	"pool_dims": (2, 2), 
}


class Net(nn.Module):
	def __init__(self, architecture_dict):
		super().__init__()

		self.kernel_size = architecture_dict["kernel_size"]
		self.padding = architecture_dict["padding"]
		self.stride = architecture_dict["stride"]
		self.pool_dims = architecture_dict["pool_dims"]

		self.encoder1 = EncoderBlock(3, 17, 2, self.kernel_size, self.padding, self.stride, self.pool_dims)
		self.encoder2 = EncoderBlock(17, 29, 2, self.kernel_size, self.padding, self.stride, self.pool_dims)
		self.encoder3 = EncoderBlock(29, 31, 3, self.kernel_size, self.padding, self.stride, self.pool_dims)

		self.decoder1 = DecoderBlock(31, 29,  3, self.kernel_size, self.padding, self.stride, self.pool_dims)
		self.decoder2 = DecoderBlock(29, 17, 2, self.kernel_size, self.padding, self.stride, self.pool_dims)
		self.decoder3 = DecoderBlock(17, 3, 2, self.kernel_size, self.padding, self.stride, self.pool_dims)


	def forward(self, x):
		x, ind1, size1 = self.encoder1(x)
		x, ind2, size2 = self.encoder2(x)
		x, ind3, size3 = self.encoder3(x)
		
		x = self.decoder1(x, ind3, size3)
		x = self.decoder2(x, ind2, size2)
		x = self.decoder3(x, ind1, size1)

		return F.softmax(x, dim=1)





#net = Net(example_architecture)
#print(net)
#x = torch.randn(10, 3, 512, 512)
#net(x)