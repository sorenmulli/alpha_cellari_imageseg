import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


class BlueLayer(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding, stride, probs = None, bias=True, dilation=1):
		super().__init__()
		
		self.convolutional = nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding, stride = stride,bias =  bias, dilation =  dilation)
		self.dropout = nn.Dropout(p = probs, inplace = False)
		self.bnorm = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU()
		self.probs = probs

	def forward(self, x):
		
		x = self.convolutional(x)
		if self.probs == None:
			pass
		else:
			x = self.dropout(x)
		x = self.bnorm(x)
		x = self.relu(x)

		return x
#https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
class EncoderBlock(nn.Module):
	def __init__(self, in_size, out_size, n_layers, kernel_size, padding, stride, mpool_dim, dropout):
		super().__init__()
		
		layers = []

		layers.append(
			BlueLayer(in_size, out_size, kernel_size, padding, stride, dropout)
		)
		
		for _ in range(n_layers-1):
			layers.append(
				BlueLayer(out_size, out_size, kernel_size, padding, stride, dropout)
			)
		
		self.encoder = nn.Sequential(*layers)

		self.mpool = nn.MaxPool2d(*mpool_dim, return_indices = True)
	
	def forward(self, x):
		x = self.encoder(x)
		upsample_size = x.size() 
		
		x, indices = self.mpool(x)

		return x, indices, upsample_size  

class DecoderBlock(nn.Module):
	def __init__(self, in_size, out_size, n_layers, kernel_size, padding, stride, unpool_dim, dropout):
		super().__init__()
		
		layers = []

		
		for _ in range(n_layers-1):
			layers.append(
				BlueLayer(in_size, in_size, kernel_size, padding, stride, dropout)
			)

		layers.append(
			BlueLayer(in_size, out_size, kernel_size, padding, stride, dropout)
		)

		self.unpool = nn.MaxUnpool2d(*unpool_dim)

		self.decoder = nn.Sequential(*layers)

	
	def forward(self, x, indices, upsample_size):
		x = self.unpool(input = x, indices = indices, output_size = upsample_size)
		x = self.decoder(x)
		return x  



		
