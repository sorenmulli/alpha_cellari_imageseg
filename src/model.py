from os.path import getsize, exists
import os 

import json
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

CPU = torch.device("cpu")
GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):

	model_fname = "/model.pt"
	json_fname = "/architecture.json"

	def __init__(self, architecture_dict, log: Logger=NullLogger()):
		super().__init__()

		self.architecture_dict = architecture_dict
		try:
			self.simple = architecture_dict["reduce_complexity"]
		except KeyError:
			self.simple = False
		self.log = log

		self.kernel_size = architecture_dict["kernel_size"]
		self.padding = architecture_dict["padding"]
		self.stride = architecture_dict["stride"]
		self.pool_dims = architecture_dict["pool_dims"]
		self.probs = architecture_dict["probs"]

		self.log("Initializing encoding blocks...")
		
		self.encoder1 = EncoderBlock(3, 64, 2, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		self.encoder2 = EncoderBlock(64, 128, 2, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		self.encoder3 = EncoderBlock(128, 256, 3, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		self.encoder4 = EncoderBlock(256, 512, 3, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		
		if not self.simple:
			self.encoder5 = EncoderBlock(512, 512, 3, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		
		self.log("Done initializing encoding blocks\n")

		self.log("Initializing decoder blocks...")
		if not self.simple:
			self.decoder1 = DecoderBlock(512, 512, 3, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		self.decoder2 = DecoderBlock(512, 256, 3, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		self.decoder3 = DecoderBlock(256, 128,  3, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		self.decoder4 = DecoderBlock(128, 64, 2, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		self.decoder5 = DecoderBlock(64, 3, 2, self.kernel_size, self.padding, self.stride, self.pool_dims, self.probs)
		self.log("Done initializing decoder blocks\n")

	def forward(self, x):

		self.log("Forwarding through encoder blocks...")
		x, ind1, size1 = self.encoder1(x)
		x, ind2, size2 = self.encoder2(x)
		x, ind3, size3 = self.encoder3(x)

		x, ind4, size4 = self.encoder4(x)
		if not self.simple:
			x, ind5, size5 = self.encoder5(x)
		self.log("Done forwarding through encoder blocks. Shape: %s" % (x.shape,))
		
		self.log("Forwarding through decoder blocks...")
		if not self.simple:
			x = self.decoder1(x, ind5, size5)
		x = self.decoder2(x, ind4, size4)
		x = self.decoder3(x, ind3, size3)
		x = self.decoder4(x, ind2, size2)
		x = self.decoder5(x, ind1, size1)

		x = F.softmax(x, dim=1)

		self.log("Done forwarding. Shape: %s" % (x.shape,))

		return x
	
	def save(self, folder: str):

		self.log("Saving model to folder %s..." % folder)
		
		os.makedirs(folder, exist_ok = True)

		model_path = folder + self.model_fname
		torch.save(self.state_dict(), model_path)
		with open(folder + self.json_fname, "w", encoding="utf-8") as f:
			json.dump(self.architecture_dict, f, indent=4)

		self.log(f"Done saving model. Size: {getsize(model_path):,} bytes")
	
	@classmethod
	def from_model(cls, folder: str, logger: Logger = NullLogger()):

		state_dict = torch.load(folder + cls.model_fname, map_location=GPU)
		with open(folder + cls.json_fname, encoding="utf-8") as f:
			architecture_dict = json.load(f)
		
		net = Net(architecture_dict, logger)
		net.load_state_dict(state_dict)
		net = net.to(GPU)

		return net



