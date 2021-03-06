#########################################################
# My job:												#
# Cut raw image/GT into a suitable number of images. 	#
# Standardize images 				 					#
# Divide images into train, eval, test, and voids.				#
# Save images in efficient format						#
#########################################################
import numpy as np
import json
import numpy as np
from PIL import Image
import wget
import os, sys, glob 

from logger import Logger

EPS = np.finfo("float64").eps

IMAGE_SHAPE = (512, 512, 3)  # Height, width, channel
EXTRA_IMAGE_SIZE = 50
IMAGE_PATHS = ("local_data/raw.png", "local_data/target.png")
CORN_PATHS = ("saved_data/cornfield_data/images", "saved_data/cornfield_data/annotated")
SUB_PATH = "local_data/imgs/{type}-{index}.png"
SPLIT = (.83, .17,)

MR_COOL_IDCS = np.array([
	0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
	1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0,
	0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
	0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
	0, 1, 0, 0, 0, 1], dtype = np.bool)

LOG = Logger("logs/data_prepper.log", "Preparation of data with images of shape %s" % ((IMAGE_SHAPE,)))
USE_NPZ = False

def _save_images():

	if not os.path.isdir("local_data"):
		os.makedirs("local_data")
	elif os.path.isfile(IMAGE_PATHS[0]) and os.path.isfile(IMAGE_PATHS[1]):
		return

	raw_image_url = 'http://www.lapix.ufsc.br/wp-content/uploads/2019/05/sugarcane2.png'
	target_image_url = 'http://www.lapix.ufsc.br/wp-content/uploads/2019/05/crop6GT.png'

	wget.download(raw_image_url, IMAGE_PATHS[0])
	wget.download(target_image_url, IMAGE_PATHS[1])

def _load_image(path: str, dtype = np.float64, channels = 3):

	"""
	Returns the pixel values of train and tests i as an m x n x 3 array
	"""

	img = Image.open(path)
	pixels = np.asarray(img, dtype = dtype)
	if channels == 3:
		pixels = pixels[:, :, :IMAGE_SHAPE[2]]
	
	return pixels

def _standardize(image, for_corn = False):

	"""
	Standardizes the pixel values of non-void pixels channelwise
	"""
 
	# Detects void pixels
	if for_corn:
		void_pixels = (image==0).all(axis=2)
	else:
		void_pixels = False

	means = list()
	stds = list()
	if for_corn:
		for i in range(image.shape[3]):
			im_channel = image[:, :, :, i]
			mean = im_channel.mean()
			std = im_channel.std() + EPS
			means.append(mean)
			stds.append(std)
			im_channel = (im_channel - mean) / std
	else:
		for i in range(image.shape[2]):
			# Singles out channel
			# Changing this will also change image, as they point to the same object
			im_channel = image[:, :, i]
			# Standardizes non-void pixels
			mean = im_channel[~void_pixels].mean()
			std = im_channel[~void_pixels].std() + EPS
			means.append(mean)
			stds.append(std)
			im_channel[~void_pixels] = (im_channel[~void_pixels] - mean) / std

	return image, means, stds

def _target_index_corn(image: np.ndarray):
	uniques = np.unique(image)
	classes = np.empty_like(image, dtype=np.uint8)
	for i, unique in enumerate(uniques):
		classes[image==unique] = i
	return classes, [f"{int(x):03}" for x in uniques]

def _target_index_sugarcane(image: np.ndarray, consider_voids = True):
	str_rep = np.apply_along_axis(lambda arr: "".join([f"{int(x):03}" for x in arr]), 2, image).astype(np.str)
	classes = np.empty(image.shape[:2], dtype=np.uint8)
	class_no = 0

	if consider_voids:
		void_class = "0" * 9
	else:
		void_class = None
	
	uniques = np.unique(str_rep)
	classes_order = np.empty(len(uniques), dtype="<U9")
	for class_ in uniques:
		if class_ != void_class:
			classes[str_rep==class_] = class_no
			classes_order[class_no] = str(class_)
			class_no += 1
		else:
			classes[str_rep==void_class] = len(uniques) - 1
			classes_order[-1] = str(void_class)
	
	return classes, [str(x) for x in classes_order]

def _pad(image, mirror_padding = False, extra_shape = 0):

	"""
	Pads an m x n x 3 array on the right and bottom such that images of shape IMAGE_SHAPE fit nicely
	"""

	height, width, channels = IMAGE_SHAPE[0] + extra_shape, IMAGE_SHAPE[1] + extra_shape, IMAGE_SHAPE[2]

	extra_height = height - image.shape[0] % height
	extra_width = width - image.shape[1] % width 
	
	if mirror_padding: 
		padded_img = np.pad(image, (extra_height, extra_width), 'reflect')
	else:
		new_dimensions = (image.shape[0] + extra_height, image.shape[1] + extra_width)
		padded_shape =  (*new_dimensions, channels) if channels else new_dimensions
		padded_img = np.zeros(padded_shape)
		padded_img[:image.shape[0], :image.shape[1]] = image
	
	return padded_img

def _split_image(image, channels, large_imgs = True):

	"""
	Splits an image into n images of shape IMAGE_SHAPE and returns them as an n_images x height x width x n_channels array
	"""

	split_shape = image.shape[0] // IMAGE_SHAPE[0],\
				image.shape[1] // IMAGE_SHAPE[1]
	n_imgs = split_shape[0] * split_shape[1]
	
	split_dim = (IMAGE_SHAPE[0], IMAGE_SHAPE[1], channels) if channels else (IMAGE_SHAPE[0], IMAGE_SHAPE[1])
	split_imgs = np.empty((n_imgs, *split_dim))

	if large_imgs:
		large_split_dim = (IMAGE_SHAPE[0]+ EXTRA_IMAGE_SIZE, IMAGE_SHAPE[1]+EXTRA_IMAGE_SIZE, channels) if channels else (IMAGE_SHAPE[0]+ EXTRA_IMAGE_SIZE, IMAGE_SHAPE[1]+EXTRA_IMAGE_SIZE)
		large_split_imgs = np.empty((n_imgs, *large_split_dim))
		large_image = _pad(image, extra_shape=EXTRA_IMAGE_SIZE // 2)

	else:
		large_split_imgs = np.empty([1])	

	for i in range(split_shape[0]):
		for j in range(split_shape[1]):
			cut = image[i*IMAGE_SHAPE[0]:(i+1)*IMAGE_SHAPE[0], j*IMAGE_SHAPE[1]:(j+1)*IMAGE_SHAPE[1]]
			split_imgs[i*split_shape[1]+j] = cut
			
			if large_imgs:
				large_cut = large_image[i*IMAGE_SHAPE[0]:(i+1)*IMAGE_SHAPE[0]+EXTRA_IMAGE_SIZE, j*IMAGE_SHAPE[1]:(j+1)*IMAGE_SHAPE[1]+EXTRA_IMAGE_SIZE]
				large_split_imgs[i*split_shape[1]+j] = large_cut


	return split_imgs, split_shape, large_split_imgs

def _find_voids(images: np.ndarray):

	"""
	Returns a boolean vector of where the images are void
	"""

	voids = np.zeros(images.shape[0], np.bool)
	for i in range(images.shape[0]):
		voids[i] = (images[i]==0).all()

	return voids

def _split_data(voids: np.ndarray, already_split = False, cutoff = None, size = None):

	"""
	voids must be a boolean vector
	Splits images into train, validation, test, and voids
	Returns four integer lists, each containing indices of images that belong to the respective category
	"""
	if already_split:
		train_val_idcs = np.arange(cutoff)
		n_train = cutoff 
		test_idcs = np.arange(cutoff, size)



	else:
		train_val_idcs = np.where(~(MR_COOL_IDCS | voids))[0]
		test_idcs = np.where(MR_COOL_IDCS & ~voids)[0]

		# Calculates size of different sets
		n_train = int(SPLIT[0] * train_val_idcs.size)
		np.random.shuffle(train_val_idcs)
	

	# Gets arrays of indices
	# Converts to list so they can be saved in json
	train_idcs = [int(x) for x in train_val_idcs[:n_train]]
	val_idcs = [int(x) for x in train_val_idcs[n_train:]]
	test_idcs = [int(x) for x in test_idcs]
	void_idcs = np.array([]) if voids is None else  [int(x) for x in np.where(voids)[0]]

	return train_idcs, val_idcs, test_idcs, void_idcs

def _prepare_data_corn():
	LOG("Loading images...")
	aerial_train = [_load_image(x) for x in glob.glob(f"{CORN_PATHS[0]}/train/*.JPG")]
	aerial_test = [_load_image(x) for x in glob.glob(f"{CORN_PATHS[0]}/test/*.JPG")]
	
	target_train = [_load_image(x, channels = 1) for x in glob.glob(f"{CORN_PATHS[1]}/train/*.png")]
	target_test = [_load_image(x, channels = 1) for x in glob.glob(f"{CORN_PATHS[1]}/test/*.png")]
	
	aerial = np.array(aerial_train + aerial_test)
	target = np.array(target_train + target_test)
	LOG("Done loading images\nShapes: %s\n train/test Split: %s\n" % (aerial.shape[1:3], (len(target_train), len(target_test))))

	LOG("Standardizing aerial image...")
	aerial, means, stds = _standardize(aerial, for_corn = True)
	LOG("Done standardizing images\n")

	LOG("Squeezing target images to single channel...")
	target, classes = _target_index_corn(target) 
	LOG("Done creating target values.\nClasses including void (if any) last:\n%s\n" % "\n".join(class_ for class_ in classes))

	LOG("Transposing images to PyTorch's preferred format...")
	aerial = np.transpose(aerial, (0, 3, 1, 2))
	LOG(f"Images transposed. Shape: {aerial.shape}\n")

	LOG("Saving images...")
	aerial_path = "local_data/aerial_prepared"
	target_path = "local_data/target_prepared"
	
	
	if USE_NPZ:
		np.savez_compressed(aerial_path, aerial.astype(np.float64))
		np.savez_compressed(target_path, target.astype(np.int8))
		

	else:
		np.save(aerial_path, aerial.astype(np.float64))
		np.save(target_path, target.astype(np.int8))


	LOG("Saved aerial images to '%s' and target images to '%s%s'\n" % (
		aerial_path,
		target_path,
		".npz" if USE_NPZ else ".npy"
	))

	LOG("Splitting images into train, validation, test, and voids...")
	train_idcs, val_idcs, test_idcs, void_idcs = _split_data(None, already_split=True, cutoff = len(target_train), size =  len(target))
	LOG("Done splitting images\nTrain: %i images\nValidation: %i images\nTest: %i images\nVoid: %i images\n"
		% (len(train_idcs), len(val_idcs), len(test_idcs), len(void_idcs)))

	LOG("Saving data preparation output...")
	prep_out = {
		"image_shape": IMAGE_SHAPE,
		"split": SPLIT,
		"means": means,
		"stds": stds,
		"aerial_path": aerial_path,
		"target_path": target_path,
		"sub_imgs_folder": SUB_PATH,
		"classes": classes,
		"train_idcs": sorted(train_idcs),
		"val_idcs": sorted(val_idcs),
		"test_idcs": sorted(test_idcs),
		"void_idcs": sorted(void_idcs),
		"extra_image_size": EXTRA_IMAGE_SIZE
	}
	json_path = "local_data/prep_out.json"
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(prep_out, f, indent=4)
	LOG("Done saving output to '%s'\n" % json_path)

	LOG("Done preparing data\n")

def _prepare_data_sugarcane():

	LOG("Downloading images...")
	_save_images()
	LOG("Done downloading images to paths\n%s\n%s\n" % IMAGE_PATHS)

	LOG("Loading images...")
	aerial, target = [_load_image(x) for x in IMAGE_PATHS]
	LOG("Done loading images\nShapes: %s\nSplit: %s\n" % (aerial.shape, SPLIT))

	LOG("Padding images...")
	aerial = _pad(aerial)
	target = _pad(target)
	LOG("Done padding images\nShapes: %s\n" % (aerial.shape,))

	LOG("Saving subimages to location %s" % SUB_PATH)
	aerial_imgs = _split_image(aerial, IMAGE_SHAPE[2])[0].astype(np.uint8)
	target_imgs = _split_image(target, IMAGE_SHAPE[2])[0].astype(np.uint8)
	for i in range(len(aerial_imgs)):
		Image.fromarray(aerial_imgs[i]).save(SUB_PATH.format(type="aerial", index=i))
		Image.fromarray(target_imgs[i]).save(SUB_PATH.format(type="target", index=i))
	LOG("Done saving subimages\n")

	LOG("Standardizing aerial image...")
	aerial, means, stds = _standardize(aerial)
	LOG("Done standardizing image\n")

	LOG("Squeezing target images to single channel...")
	target, classes = _target_index_sugarcane(target)
	LOG("Done creating target values.\nClasses including void (if any) last:\n%s\n" % "\n".join(class_ for class_ in classes))

	LOG("Splitting images...")
	aerial, split_shape, large_aerial = _split_image(aerial, IMAGE_SHAPE[2])
	target, _, large_target = _split_image(target, None, False)
	LOG("Done splitting images\nNumber of images: %i\nShapes: %s\n" % (aerial.shape[0], IMAGE_SHAPE))

	LOG("Detecting void images...")
	voids = _find_voids(aerial)
	LOG("Done finding voids\n%i images where voids\n" % voids.sum())

	LOG("Transposing images to PyTorch's preferred format...")
	aerial = np.transpose(aerial, (0, 3, 1, 2))
	large_aerial = np.transpose(large_aerial, (0, 3, 1, 2))
	LOG(f"Images transposed. Shape: {aerial.shape}\n")

	LOG("Saving images...")
	aerial_path = "local_data/aerial_prepared"
	target_path = "local_data/target_prepared"
	
	large_aerial_path = "local_data/large_aerial_prepared"
	large_target_path = "local_data/large_target_prepared"
	
	if USE_NPZ:
		np.savez_compressed(aerial_path, aerial.astype(np.float64))
		np.savez_compressed(target_path, target.astype(np.int8))
		
		np.savez_compressed(large_aerial_path, large_aerial.astype(np.float64))
		np.savez_compressed(large_target_path, large_target.astype(np.int8))

	else:
		np.save(aerial_path, aerial.astype(np.float64))
		np.save(target_path, target.astype(np.int8))

		np.save(large_aerial_path, large_aerial.astype(np.float64))
		np.save(large_target_path, large_target.astype(np.int8))

	LOG("Saved aerial images to '%s' and target images to '%s%s'\n" % (
		aerial_path,
		target_path,
		".npz" if USE_NPZ else ".npy"
	))

	LOG("Splitting images into train, validation, test, and voids...")
	train_idcs, val_idcs, test_idcs, void_idcs = _split_data(voids)
	LOG("Done splitting images\nTrain: %i images\nValidation: %i images\nTest: %i images\nVoid: %i images\n"
		% (len(train_idcs), len(val_idcs), len(test_idcs), len(void_idcs)))

	LOG("Saving data preparation output...")
	prep_out = {
		"image_shape": IMAGE_SHAPE,
		"split_shape": split_shape,
		"split": SPLIT,
		"means": means,
		"stds": stds,
		"aerial_path": aerial_path,
		"target_path": target_path,
		"large_aerial_path": large_aerial_path,
		"large_target_path": large_target_path,
		"sub_imgs_folder": SUB_PATH,
		"classes": classes,
		"train_idcs": sorted(train_idcs),
		"val_idcs": sorted(val_idcs),
		"test_idcs": sorted(test_idcs),
		"void_idcs": sorted(void_idcs),
		"extra_image_size": EXTRA_IMAGE_SIZE
	}
	json_path = "local_data/prep_out.json"
	with open(json_path, "w", encoding="utf-8") as f:
		json.dump(prep_out, f, indent=4)
	LOG("Done saving output to '%s'\n" % json_path)

	LOG("Done preparing data\n")


if __name__ == "__main__":
	os.chdir(sys.path[0])
	os.makedirs("local_data/imgs", exist_ok = True)

	_prepare_data_sugarcane()
	#_prepare_data_corn()