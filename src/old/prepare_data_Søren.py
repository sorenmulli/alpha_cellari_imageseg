#########################################################
# My job:												#
# Cut raw image/GT into a suitable number of images. 	#
# Divide images into train/eval and test.				#
# Save images in efficient format						#
# Perform pre-processing if necessary 					#
#########################################################
import numpy as np

import os, sys
os.chdir(sys.path[0])

from imageio import imread
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings("ignore", message="Data with input dtype uint8 was converted to float64 by StandardScaler.")

SINGLE_SHAPE = (512, 512)
CHANNELS = 3

def prepare(path):
	#Load and throw away dimension without information
	GT = imread(f'{path}raw_GT.png') 
	data = imread(f'{path}raw_image.png')[:, :, :CHANNELS]
	

	#Standardize
	scaler = StandardScaler()

	stan_data = np.empty_like(data)
	for channel in range(CHANNELS):
		stan_data[:, :, channel] = scaler.fit_transform(data[:, :, channel]) 
	
	stan_data = stan_data.astype(np.uint8)

	assert GT.shape == data.shape

	print(f"Data dimensionality: {data.shape}")
	return GT, stan_data	

def pad(data, single_img_shape):
	(height, width) = data.shape[:2]

	to_pad = (height % single_img_shape[0], width % single_img_shape[1])
	
	#print(height + to_pad[0])
	#print(width + to_pad[1])
	#data = np.pad(data, to_pad, 'constant')
	
	padded_size = (height + height % single_img_shape[0], width + width % single_img_shape[1] )
	padded_data = np.zeros((*padded_size, CHANNELS))

	for channel in range(CHANNELS):
		padded_data[:height, :width, channel] =  data[:, :, channel]
	return padded_data



if __name__ == "__main__":
	GT, data = prepare('data/')
	
	GT = pad(GT, SINGLE_SHAPE)
	data = pad(data, SINGLE_SHAPE)

