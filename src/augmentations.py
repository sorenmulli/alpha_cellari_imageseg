from dataclasses import dataclass

import numpy as np
import torch

from logger import Logger, NullLogger

@dataclass
class AugmentationConfig:
	cropsize = (450, 450)
	augment_p: list
	augments: list


class Augmenter:

	def __init__(self, augment_cfg: AugmentationConfig=AugmentationConfig([], []), logger: Logger=NullLogger()):

		self.augment_cfg = augment_cfg

		self.log = logger
	
	def _crop(self, image, target):

		"""
		Crops image and target base self.augment_cfg.cropsize
		Input size: n_images x channels x height x width
		"""

		assert image.shape == target.shape

		h_shift = np.random.randint(image.shape[2]-self.augment_cfg.cropsize[0])
		w_shift = np.random.randint(target.shape[3]-self.augment_cfg.cropsize[1])

		image = image[:, :, h_shift:, w_shift:]
		target = target[:, :, h_shift, w_shift:]

		return image, target
	
	def augment(self, image, target):
		
		self.log("Cropping images...")
		image, target = self._crop(image, target)
		self.log("Done cropping images. Shape: %s\n" % (image.shape,))

		self.log("Appyling augmentations...")
		for i, augmentation in enumerate(self.augment_cfg.augments):
			if self.augment_cfg.augment_p[i] > np.random.random():
				self.log("Applying augmentation %s..." % augmentation)
				image, target = augmentation(image, target)
		self.log("Done applying augmentations. Shape: %s\n" % (image.shape,))
		
		return image, target

def _flip_tb(image: torch.tensor, target: torch.tensor):

	"""
	Flips image and target images along the horizontal axis
	Input size: n_images x channels x height x width
	"""

	assert image.shape == target.shape

	image = image.flip(3)

def _flip_lr(image: torch.tensor, target: torch.tensor):

	"""
	Flips image and target images along the horizontal axis
	Input size: n_images x channels x height x width
	"""

	assert image.shape == target.shape

	image = image.flip(4)

	return image


if __name__ == "__main__":

	# Test case
	from data_loader import DataLoader
	logger = Logger("logs/augmentation-test.log", "Testing image augmentations")
	img, target = DataLoader(5, Augmenter(logger=logger).augment).get_test()



