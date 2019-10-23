import os, sys
os.chdir(sys.path[0])

import json
import numpy as np
from PIL import Image

from logger import Logger

EPS = np.finfo("float64").eps

IMAGE_SHAPE = (512, 512, 3)  # Height, width, channel
IMAGE_PATHS = ("data/raw_image.png", "data/raw_GT.png")
SPLIT = (.6, .2, .2)
assert sum(SPLIT) == 1
LOG = Logger("logs/data_prepper.log", "Preparation of data with image shape %s" % ((IMAGE_SHAPE,)))

def _load_image(path: str):

  """
  Returns the pixel values of an image as an m x n x 3 array
  """
  
  img = Image.open(path)
  pixels = np.asarray(img, np.float64)
  pixels = pixels[:, :, :IMAGE_SHAPE[2]]
  
  return pixels

def _pad(image: np.ndarray):

  """
  Pads an m x n x 3 array on the right and bottom such that an images of shape IMAGE_SHAPE fits nicely
  """

  extra_height = IMAGE_SHAPE[0] - image.shape[0] % IMAGE_SHAPE[0]
  extra_width = IMAGE_SHAPE[1] - image.shape[1] % IMAGE_SHAPE[1]
  padded_shape = (image.shape[0] + extra_height, image.shape[1] + extra_width, IMAGE_SHAPE[2])
  padded_img = np.zeros(padded_shape)
  padded_img[:image.shape[0], :image.shape[1]] = image

  return padded_img

def _split_image(image: np.ndarray):

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

def _find_voids(images: np.ndarray):

  """
  Returns a boolean vector of which images are void
  """

  voids = np.zeros(images.shape[0], np.bool)
  for i in range(images.shape[0]):
    voids[i] = (images[i]==0).all()
  
  return voids

def _standardize(images: np.ndarray, voids: np.ndarray):

  """
  Standardizes the pixel values of non-void images channelwise
  """

  for i in range(images.shape[3]):
    images[~voids, :, :, i] = (images[~voids, :, :, i] - images[~voids, :, :, i].mean()) /\
                              (images[~voids, :, :, i].std() + EPS)
  
  return images

def _split_data(voids: np.ndarray):

  """
  Splits images into train, validation, test, and voids
  Returns four integer lists, each containing indices of images that belong to the respective category
  """

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

def _prepare_data():

  LOG.log("Loading images...")
  aerial, target = [_load_image(x) for x in IMAGE_PATHS]
  LOG.log("Done loading images\nShapes: %s\nSplit: %s\n" % (aerial.shape, SPLIT))

  LOG.log("Padding images...")
  aerial, target = _pad(aerial), _pad(target)
  LOG.log("Done padding images\nShapes: %s\n" % (aerial.shape,))

  LOG.log("Splitting images...")
  aerial, target = _split_image(aerial), _split_image(target)
  LOG.log("Done splitting images.\nNumber of images: %i\nShapes: %s\n" % (aerial.shape[0], IMAGE_SHAPE))

  LOG.log("Detecting void images...")
  voids = _find_voids(aerial)
  LOG.log("Done finding voids\n2 x %i images where voids\n" % voids.sum())

  LOG.log("Standardizing images...")
  aerial, target = _standardize(aerial, voids), _standardize(target, voids)
  LOG.log("Done standardizing images\n")

  LOG.log("Saving images...")
  aerial_path = "data/aerial_prepared.npy"
  target_path = "data/target_prepared.npy"
  np.save(aerial_path, aerial.astype(np.float64))
  np.save(target_path, target.astype(np.float64))
  LOG.log("Saved aerial images to '%s.npy' and target images to '%s.npy'\n" % (aerial_path, target_path))

  LOG.log("Splitting images into train, validation, test, and voids...")
  train_idcs, val_idcs, test_idcs, void_idcs = _split_data(voids)
  LOG.log("Done splitting images.\nTrain: %i images\nValidation: %i images\nTest: %i images\nVoid: %i images\n"
          % (len(train_idcs), len(val_idcs), len(test_idcs), len(void_idcs)))

  LOG.log("Saving data preparation output...")
  prep_out = {
    "image_shape": IMAGE_SHAPE,
    "split": SPLIT,
    "aerial_path": aerial_path,
    "target_path": target_path,
    "train_idcs": train_idcs,
    "val_idcs": val_idcs,
    "test_idcs": test_idcs,
    "void_idcs": void_idcs,
  }
  json_path = "data/prep_out.json"
  with open(json_path, "w", encoding="utf-8") as f:
    json.dump(prep_out, f, indent=4)
  LOG.log("Done saving output to '%s'\n" % json_path)

  LOG.log("Done preparing data\n")


if __name__ == "__main__":

  _prepare_data()

