from os.path import getsize

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import EncoderBlock, DecoderBlock
from logger import Logger, NullLogger

example_architecture = {
	"kernel_size":  3,
	"padding": 1, 
	"stride": 1,
	"pool_dims": (2, 2), 
	"probs": 0.25,
}


class Net(nn.Module):
	def __init__(self, architecture_dict, log: Logger=NullLogger()):
		super().__init__()

		self.log = log

		self.kernel_size = architecture_dict["kernel_size"]
		self.padding = architecture_dict["padding"]
		self.stride = architecture_dict["stride"]
		self.pool_dims = architecture_dict["pool_dims"]
		self.probs = architecture_dict["probs"]

		self.log("Initializing encoding blocks...")
		self.encoder1 = EncoderBlock(3, 32, 2, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		self.encoder2 = EncoderBlock(32, 64, 2, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		self.encoder3 = EncoderBlock(64, 128, 3, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		self.log("Done initializing encoding blocks\n")

		self.log("Initializing decoder blocks...")
		self.decoder1 = DecoderBlock(128, 64,  3, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		self.decoder2 = DecoderBlock(64, 32, 2, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		self.decoder3 = DecoderBlock(32, 3, 2, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		self.log("Done initializing decoder blocks\n")

	def forward(self, x):

		self.log("Forwarding through encoder blocks...")
		x, ind1, size1 = self.encoder1(x)
		x, ind2, size2 = self.encoder2(x)
		x, ind3, size3 = self.encoder3(x)
		self.log("Done forwarding through encoder blocks. Shape: %s" % (x.shape,))
		
		self.log("Forwarding through decoder blocks...")
		x = self.decoder1(x, ind3, size3)
		x = self.decoder2(x, ind2, size2)
		x = self.decoder3(x, ind1, size1)
		x = F.softmax(x, dim=1)
		self.log("Done forwarding. Shape: %s" % (x.shape,))

		return x
	
	def save(self, path: str):

		self.log("Saving model to path %s..." % path)
		torch.save(self.state_dict(), path)
		self.log(f"Done saving model. Size: {getsize(path):,} bytes")





#net = Net(example_architecture)
#print(net)
#x = torch.randn(10, 3, 512, 512)
#net(x)
#net.save("test_model.pt")