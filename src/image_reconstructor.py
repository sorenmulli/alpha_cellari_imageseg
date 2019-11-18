import os, sys

import json
import numpy as np
from PIL import Image

from logger import Logger, NullLogger
from data_loader import DataLoader


def ensure_shape(arr: np.ndarray or torch.Tensor, axes = 4):

	"""
	Ensures an array has four axes. If it has three, a fourth axis is added as the first, so the shape is 1 x m x n x o
	"""

	if len(arr.shape) == axes - 1:
		arr = arr.reshape((1, *arr.shape))
	
	return arr

class ImageReconstructor:

	def __init__(self, logger: Logger = NullLogger(), json_path: str = "local_data/prep_out.json"):
		with open(json_path , encoding="utf-8") as f:
			self.cfg = json.load(f)
		self.means = self.cfg["means"]
		self.stds = self.cfg["stds"]
		self.paths = self.cfg["aerial_path"], self.cfg["target_path"]

		self.log = logger
		
	def _reconstruct_aerial(self, standardized: np.ndarray):

		"""
		Reconstructs standardized images from saved mean and stds
		standardized should have shape n_imgs x width x height x channels
		"""

		void_pixels = (standardized==0).all(axis=1)
		for i in range(standardized.shape[1]):
			channel = standardized[:, i, :, :]
			channel[~void_pixels] = channel[~void_pixels] * self.stds[i] + self.means[i]
		images = standardized.astype(np.uint8)

		return np.transpose(images, (0, 2, 3, 1))

	def reconstruct_aerial(self, standardized: np.ndarray, *show):

		"""
		Reconstructs images from standardized images
		Should be of shape width x height x channels for single image or n x width x height x channels for n images
		to_show: Indices of images that should be showed when reconstructed if show
		"""

		# Makes sure the standardized array has four axes
		standardized = ensure_shape(standardized)

		# Does the actual image reconstruction
		self.log("Starting reconstruction of %i aerial image(s)..." % len(standardized))
		aerial = self._reconstruct_aerial(standardized)
		self.log("Done reconstructing %i aerial images\n" % len(standardized))

		# Shows demanded reconstructions
		if len(show) != 0:
			self.log("Showing reconstructed %i image(s)\n" % len(show))
		else:
			self.log("Not showing any images\n")
		
		self.log("Showing %i images\n" % len(show))
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

	def reconstruct_output(self, output: np.ndarray, voids: np.ndarray, *show):

		"""
		Reconstructs output image from network
		output shape: n x channels x height x width
		yellow = np.array([255, 255, 0], dtype=np.uint8)_images x n_channels x height x width
		voids: Boolean vector of shape n_images x height x width
		"""

		self.log("Ensuring shape...")
		output = ensure_shape(output)
		voids = ensure_shape(voids, axes=3)
		output = output.transpose(0, 2, 3, 1)
		self.log("Done ensuring shape. Shape: %s\n" % (output.shape, ))

		red = np.array([255, 0, 0], dtype=np.uint8)
		green = np.array([0, 255, 0], dtype=np.uint8)
		yellow = np.array([255, 255, 0], dtype=np.uint8)
		black = np.array([0, 0, 0], dtype=np.uint8)

		self.log("Determining classes and inserting colours...")
		classes = np.argmax(output, axis=3)
		reconst = np.zeros_like(output, dtype=np.uint8)
		reconst[classes==0] = red
		reconst[classes==1] = green
		reconst[classes==2] = yellow
		if voids is not None:
			self.log("Setting void pixels to black...")
			reconst[voids] = black
		self.log("Done determining classes and inserting colours\n")

		self.log("Showing %i images\n" % len(show))
		for i in show:
			Image.fromarray(reconst[i]).show()

		return reconst
		

if __name__ == "__main__":
	os.chdir(sys.path[0])

	# Test cases
	reconstructor = ImageReconstructor(Logger("logs/test-reconstruction.log", "Reconstructing images from data"))
	reconstructor.reconstruct_aerial_from_file(reconstructor.paths[0], 3, 5, 9)

	voids = np.random.randint(2, size=(2, 512, 512), dtype=np.bool)
	test_output = np.random.randn(2, 3, 512, 512)
	reconstructor.reconstruct_output(test_output, voids, 0, 1)




