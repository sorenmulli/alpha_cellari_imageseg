import os, sys

import json
from PIL import Image

import numpy as np
import torch

from logger import Logger, NullLogger
	
def stitch(images: np.ndarray, split_shape: tuple, show = False, savepath: str = ""):

	"""
	Performs stitching of reconstructed numpy images
	Input shape: n_images x height x width x n_channels
	split_shape is the shape of the chopping of the original image, saved in local_data/prep_out.json
	(n_vstacked, n_hstacked)
	"""
	
	# Calculates dimensions of total image
	height = images.shape[1] * split_shape[0]
	width = images.shape[2] * split_shape[1]
	img = np.empty((height, width, 3), dtype=np.uint8)
	# Reshapes images to single images
	for i in range(split_shape[0]):
		for j in range(split_shape[1]):
			idx = i * split_shape[1] + j
			img[i*images.shape[1]:(i+1)*images.shape[1], j*images.shape[2]:(j+1)*images.shape[2]] = images[idx]

	if show:
		Image.fromarray(img).show()
	if savepath:
		Image.fromarray(img).save(savepath)
	
	return img

		
if __name__ == "__main__":
	os.chdir(sys.path[0])

	# Test case
	from image_reconstructor import ImageReconstructor
	imgs = ImageReconstructor().reconstruct_aerial_from_file("local_data/aerial_prepared")
	img = stitch(imgs, (14, 11), True, "local_data/test-stitch.png")




