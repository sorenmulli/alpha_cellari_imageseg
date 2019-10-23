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

IMAGE_SHAPE = (512, 512)
CHANNELS = 3
SPLIT = (.7, .2, .1)

assert sum(SPLIT) == 1

def read(path):
	#Load and throw away dimension without information
	GT = imread(f'{path}raw_GT.png') 
	data = imread(f'{path}raw_image.png')[:, :, :CHANNELS]

	assert GT.shape == data.shape

	print(f"Data dimensionality: {data.shape}")
	return GT, data	

def pad(data, single_img_shape):
	(height, width) = data.shape[:2]
	
	padded_size = (height + height % single_img_shape[0], width + width % single_img_shape[1])
	padded_data = np.zeros((*padded_size, CHANNELS))

	for channel in range(CHANNELS):
		padded_data[:height, :width, channel] =  data[:, :, channel]
	return padded_data

def find_voids(images):
	voids = np.zeros(images.shape[0], np.bool)
	for i in range(images.shape[0]):
		voids[i] = (images[i]==0).all()

	return voids

def standardize(images, voids):
	scaler = StandardScaler()

	for i in range(images.shape[3]):
		images[~voids, :, :, i] = scaler.fit_transform(images[~voids, :, :, i])

	return images

def partition(data):

	split_shape = data.shape[0] // IMAGE_SHAPE[0],\
			data.shape[1] // IMAGE_SHAPE[1]
	n_imgs = split_shape[0] * split_shape[1]
	split_imgs = np.empty((n_imgs, *IMAGE_SHAPE))
	for i in range(split_shape[0]):
		for j in range(split_shape[1]):
			cut = data[i*IMAGE_SHAPE[0]:(i+1)*IMAGE_SHAPE[0], j*IMAGE_SHAPE[1]:(j+1)*IMAGE_SHAPE[1]]
			split_imgs[i*split_shape[1]+j] = cut
  
	return split_imgs


def shuffle(voids):
	# Generates all indices, removes those that are void, and shuffles
	idcs = np.arange(voids.size)
	idcs = idcs[~voids]
	np.random.shuffle(idcs)

	# Calculates size of different sets
	n_train = int(SPLIT[0] * idcs.size)
	n_val = int(SPLIT[1] * idcs.size)
	n_test = int(SPLIT[2] * idcs.size)

	# Gets arrays of indices
	# Converts to list so they can be saved in json
	train_idcs = list(int(x) for x in idcs[:n_train])
	val_idcs = list(int(x) for x in idcs[n_train:n_train+n_val])
	test_idcs = list(int(x) for x in idcs[n_train+n_val:n_train+n_val+n_test])
	void_idcs = list(int(x) for x in np.where(voids)[0])

	return train_idcs, val_idcs, test_idcs, void_idcs



if __name__ == "__main__":
	GT, data = read('data/')

	GT = pad(GT, IMAGE_SHAPE)
	data = pad(data, IMAGE_SHAPE)


