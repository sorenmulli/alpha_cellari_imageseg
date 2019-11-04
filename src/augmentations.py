from dataclasses import dataclass

import numpy as np
import torch

from logger import Logger, NullLogger

@dataclass
class AugmentationConfig:
	cropsize = (450, 450)
	augments: list
	augment_p: list


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

		h, w = self.augment_cfg.cropsize
		h_shift = np.random.randint(image.shape[2]-h)
		w_shift = np.random.randint(target.shape[3]-w)

		image = image[:, :, h_shift:h_shift+h, w_shift:w_shift+w]
		target = target[:, :, h_shift:h_shift+h, w_shift:w_shift+w]

		return image, target
	
	def augment(self, image, target):
		
		self.log("Cropping images...")
		image, target = self._crop(image, target)
		self.log("Done cropping images. Shape: %s\n" % (image.shape,))

		self.log("Appyling augmentations...")
		for augmentation, p in zip(self.augment_cfg.augments, self.augment_cfg.augment_p):
			if p > np.random.random():
				self.log("Applying augmentation %s..." % augmentation)
				image, target = augmentation(image, target)
		self.log("Done applying augmentations. Shape: %s\n" % (image.shape,))
		
		return image, target

def flip_tb(image: torch.tensor, target: torch.tensor):

	"""
	Flips image and target images along the horizontal axis
	Input size: n_images x channels x height x width
	"""

	assert image.shape == target.shape

	image = image.flip(2)
	target = target.flip(2)

	return image, target

def flip_lr(image: torch.tensor, target: torch.tensor):

	"""
	Flips image and target images along the horizontal axis
	Input size: n_images x channels x height x width
	"""

	assert image.shape == target.shape

	image = image.flip(3)
	target = target.flip(3)

	return image, target


if __name__ == "__main__":

	# Test case
	from data_loader import DataLoader
	log = Logger("logs/augmentation-test.log", "Testing image augmentations")
	aug_cfg = AugmentationConfig([flip_tb, flip_lr], [.5, .5])
	img, target = DataLoader(
		"local_data/prep_out.json",
		5,
		Augmenter(aug_cfg, logger=log).augment,
		logger=log
	).get_test()



