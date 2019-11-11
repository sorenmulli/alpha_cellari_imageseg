import os, sys
os.chdir(sys.path[0])

import json
import numpy as np
from PIL import Image

from logger import Logger, NullLogger
from data_loader import DataLoader

JSON_PATH = "local_data/prep_out.json"
with open(JSON_PATH, encoding="utf-8") as f:
	CFG = json.load(f)
MEANS = CFG["means"]
STDS = CFG["stds"]
PATHS = ("local_data/aerial_prepared", "local_data/target_prepared")

class ImageReconstructor:

	def __init__(self, logger: Logger = NullLogger()):
		
		self.log = logger

	def _ensure_shape(self, arr: np.ndarray):

		"""
		Ensures an array has four axes. If it has three, a fourth axis is added as the first, so the shape is 1 x m x n x o
		"""

		if len(arr.shape) == 3:
			arr = arr.reshape((1, *arr.shape))
		
		return arr

	def _reconstruct_aerial(self, standardized: np.ndarray):

		"""
		Reconstructs standardized images from saved mean and stds
		standardized should have shape n_imgs x width x height x channels
		"""

		void_pixels = (standardized==0).all(axis=1)
		for i in range(standardized.shape[1]):
			channel = standardized[:, i, :, :]
			channel[~void_pixels] = channel[~void_pixels] * STDS[i] + MEANS[i]
		images = standardized.astype(np.uint8)

		return np.transpose(images, (0, 2, 3, 1))

	def reconstruct_aerial(self, standardized: np.ndarray, *show):

		"""
		Reconstructs images from standardized images
		Should be of shape width x height x channels for single image or n x width x height x channels for n images
		to_show: Indices of images that should be showed when reconstructed if show
		"""

		# Makes sure the standardized array has four axes
		standardized = self._ensure_shape(standardized)

		# Does the actual image reconstruction
		self.log("Starting reconstruction of %i aerial image(s)..." % len(standardized))
		aerial = self._reconstruct_aerial(standardized)
		self.log("Done reconstructing %i aerial images\n" % len(standardized))

		# Shows demanded reconstructions
		if len(show) != 0:
			self.log("Showing reconstructed %i image(s)\n" % len(show))
		else:
			self.log("Not showing any images\n")
		
		for i in show:
			im = Image.fromarray(aerial[i])
			im.show()
		
		return aerial


	def reconstruct_aerial_from_file(self, path, *show):

		"""
		Reconstructs images from saved npz's
		"""

		self.log("Loading standardized image...")
		aerial = DataLoader.load(path)
		self.log("Done loading image\n")

		aerial = self.reconstruct_aerial(aerial, *show)

		return aerial

if __name__ == "__main__":

	ImageReconstructor(
		Logger("logs/reconstruction.log", "Reconstructing images from data")
	).reconstruct_aerial_from_file(PATHS[0], 3, 5, 9)
