import os, sys
os.chdir(sys.path[0])

import json
import numpy as np
from PIL import Image

from logger import Logger

LOG = Logger("logs/reconstruction.log", "Image reconstruction")
with open("local_data/prep_out.json", encoding="utf-8") as f:
	PREPOUT = json.load(f)
MEANS = PREPOUT["means"]
STDS = PREPOUT["stds"]
PATHS = ("local_data/aerial_prepared.npz", "local_data/target_prepared.npz")

def load_npz(path: str):

	"""
	Loads a numpy array saved as a .npz file
	"""

	arr = np.load(path)["arr_0"]

	return arr

def _ensure_shape(arr: np.ndarray):

	"""
	Ensures an array has four axes. If it has three, a fourth axis is added as the first, so the shape is 1 x m x n x o
	"""

	if len(arr.shape) == 3:
		arr = arr.reshape((1, *arr.shape))
	
	return arr

def _reconstruct_aerial(standardized: np.ndarray):

	"""
	Reconstructs standardized images from saved mean and stds
	standardized should have shape n_imgs x width x height x channels
	"""

	void_pixels = (standardized==0).all(axis=1)
	for i in range(standardized.shape[1]):
		channel = standardized[:, :, :, i]
		channel[~void_pixels] = channel[~void_pixels] * STDS[i] + MEANS[i]
	images = standardized.astype(np.uint8)

	return images

def reconstruct_aerial(standardized: np.ndarray, *show):

	"""
	Reconstructs images from standardized images
	Should be of shape width x height x channels for single image or n x width x height x channels for n images
	to_show: Indices of images that should be showed when reconstructed if show
	"""

	# Makes sure the standardized array has four axes
	standardized = _ensure_shape(standardized)

	# Does the actual image reconstruction
	LOG.log("Starting reconstruction of %i aerial image(s)..." % len(standardized))
	aerial = _reconstruct_aerial(standardized)
	LOG.log("Done reconstructing %i aerial images\n" % len(standardized))

	# Shows demanded reconstructions
	if len(show) != 0:
		LOG.log("Showing reconstructed %i image(s)\n" % len(show))
	else:
		LOG.log("Not showing any images\n")
	for i in show:
		im = Image.fromarray(aerial[i])
		im.show()
	
	return aerial


def reconstruct_aerial_from_file(path, *show):

	"""
	Reconstructs images from saved npz's
	"""

	LOG.log("Loading standardized image...")
	aerial = load_npz(path)
	LOG.log("Done loading image\n")

	aerial = reconstruct_aerial(aerial, *show)

	return aerial

if __name__ == "__main__":

	reconstruct_aerial_from_file(PATHS[0])
