#########################################################
# My job:												#
# Cut raw image/GT into a suitable number of images. 	#
# Standardize images 				 					#
# Divide images into train/eval and test.				#
# Save images in efficient format						#
#########################################################
import os, sys
os.chdir(sys.path[0])

import json
import numpy as np
from PIL import Image
import wget

from logger import Logger

EPS = np.finfo("float64").eps

IMAGE_SHAPE = (512, 512, 3)  # Height, width, channel
IMAGE_PATHS = ("local_data/raw.png", "local_data/target.png")
SPLIT = (.8, .2,)

MR_COOL_IDCS = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
	1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
	0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
	0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
	0, 1, 0, 0, 0, 1], dtype=np.bool)

LOG = Logger("logs/data_prepper.log", "Preparation of data with image shape %s" % ((IMAGE_SHAPE,)))

def _save_images():

	if not os.path.isdir("local_data"):
		os.makedirs("local_data")
	elif os.path.isfile(IMAGE_PATHS[0]) and os.path.isfile(IMAGE_PATHS[1]):
		return

	raw_image_url = 'http://www.lapix.ufsc.br/wp-content/uploads/2019/05/sugarcane2.png'
	target_image_url = 'http://www.lapix.ufsc.br/wp-content/uploads/2019/05/crop6GT.png'

	wget.download(raw_image_url, IMAGE_PATHS[0])
	wget.download(target_image_url, IMAGE_PATHS[1])

def _load_image(path: str):

	"""
	Returns the pixel values of train and tests i as an m x n x 3 array
	"""

	img = Image.open(path)
	pixels = np.asarray(img, np.float64)
	pixels = pixels[:, :, :IMAGE_SHAPE[2]]

	return pixels

def _standardize(image):

	"""
	Standardizes the pixel values of non-void pixels channelwise
	"""

	# Detects void pixels
	void_pixels = (image==0).all(axis=2)
	for i in range(image.shape[2]):
		# Singles out channel
		# Changing this will also change image, as they point to the same object
		im_channel = image[:, :, i]
		# Standardizes non-void pixels
		mean = im_channel[~void_pixels].mean()
		std = im_channel[~void_pixels].std() + EPS
		im_channel[~void_pixels] = (im_channel[~void_pixels] - mean) / std

	return image

def _pad(image):

	"""
	Pads an m x n x 3 array on the right and bottom such that images of shape IMAGE_SHAPE fit nicely
	"""

	extra_height = IMAGE_SHAPE[0] - image.shape[0] % IMAGE_SHAPE[0]
	extra_width = IMAGE_SHAPE[1] - image.shape[1] % IMAGE_SHAPE[1]
	padded_shape = (image.shape[0] + extra_height, image.shape[1] + extra_width, IMAGE_SHAPE[2])
	padded_img = np.zeros(padded_shape)
	padded_img[:image.shape[0], :image.shape[1]] = image

	return padded_img

def _split_image(image):

	"""
	Splits an image into n images of shape IMAGE_SHAPE and returns them as a n x m x o x 3 array
	"""

	split_shape = image.shape[0] // IMAGE_SHAPE[0],\
				image.shape[1] // IMAGE_SHAPE[1]
	n_imgs = split_shape[0] * split_shape[1]
	split_imgs = np.empty((n_imgs, *IMAGE_SHAPE))
	for i in range(split_shape[0]):
		for j in range(split_shape[1]):
			cut = image[i*IMAGE_SHAPE[0]:(i+1)*IMAGE_SHAPE[0], j*IMAGE_SHAPE[1]:(j+1)*IMAGE_SHAPE[1]]
			split_imgs[i*split_shape[1]+j] = cut

	return split_imgs

def _find_voids(images):

	"""
	Returns a boolean vector of where the images are void
	"""

	voids = np.zeros(images.shape[0], np.bool)
	for i in range(images.shape[0]):
		voids[i] = (images[i]==0).all()

	return voids

def _split_data(voids):

	"""
	Splits images into train, validation, test, and voids
	Returns four integer lists, each containing indices of images that belong to the respective category
	"""

	train_val_idcs = np.where(~MR_COOL_IDCS)[0]
	test_idcs = np.where(MR_COOL_IDCS)[0]

	# Calculates size of different sets
	n_train = int(SPLIT[0] * train_val_idcs.size)
	np.random.shuffle(train_val_idcs)

	# Gets arrays of indices
	# Converts to list so they can be saved in json
	void_idcs = [int(x) for x in np.where(voids)[0]]
	train_idcs = [int(x) for x in train_val_idcs[:n_train] if x not in void_idcs]
	val_idcs = [int(x) for x in train_val_idcs[n_train:] if x not in void_idcs]
	test_idcs = [int(x) for x in test_idcs if x not in void_idcs]

	return train_idcs, val_idcs, test_idcs, void_idcs

def _prepare_data():

	LOG.log("Downloading images...")
	_save_images()
	LOG.log("Done downloading images to paths\n%s\n%s\n" % IMAGE_PATHS)

	LOG.log("Loading images...")
	aerial, target = [_load_image(x) for x in IMAGE_PATHS]
	LOG.log("Done loading images\nShapes: %s\nSplit: %s\n" % (aerial.shape, SPLIT))

	LOG.log("Standardizing images...")
	aerial, target = _standardize(aerial), _standardize(target)
	LOG.log("Done standardizing images\n")

	LOG.log("Padding images...")
	aerial, target = _pad(aerial), _pad(target)
	LOG.log("Done padding images\nShapes: %s\n" % (aerial.shape,))

	LOG.log("Splitting images...")
	aerial, target = _split_image(aerial), _split_image(target)
	LOG.log("Done splitting images\nNumber of images: %i\nShapes: %s\n" % (aerial.shape[0], IMAGE_SHAPE))

	LOG.log("Detecting void images...")
	void_idcs = _find_voids(aerial)
	LOG.log("Done finding voids\n2 x %i images where voids\n" % void_idcs.sum())

	LOG.log("Saving images...")
	aerial_path = "local_data/aerial_prepared.npz"
	target_path = "local_data/target_prepared.npz"
	np.savez_compressed(aerial_path, aerial.astype(np.float64))
	np.savez_compressed(target_path, target.astype(np.float64))
	LOG.log("Saved aerial images to '%s' and target images to '%s'\n" % (aerial_path, target_path))

	LOG.log("Splitting images into train, validation, test, and voids...")
	train_idcs, val_idcs, test_idcs, void_idcs = _split_data(void_idcs)
	LOG.log("Done splitting images\nTrain: %i images\nValidation: %i images\nTest: %i images\nVoid: %i images\n"
		% (len(train_idcs), len(val_idcs), len(test_idcs), len(void_idcs)))

	LOG.log("Saving data preparation output...")
	prep_out = {
		"image_shape": IMAGE_SHAPE,
		"split": SPLIT,
		# "mean": mean,  TODO
		# "std": std,
		"aerial_path": aerial_path,
		"target_path": target_path,
		"train_idcs": train_idcs,
		"val_idcs": val_idcs,
		"test_idcs": test_idcs,
		"void_idcs": void_idcs,
	}
	json_path = "prep_out.json"
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(prep_out, f, indent=4)
	LOG.log("Done saving output to '%s'\n" % json_path)

	LOG.log("Done preparing data\n")


if __name__ == "__main__":

	_prepare_data()

