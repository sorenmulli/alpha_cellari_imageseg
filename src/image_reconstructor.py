import os, sys
os.chdir(sys.path[0])

import json
import numpy as np
from PIL import Image

from logger import Logger

LOG = Logger("local_data/reconstruct.log", "Image reconstruction")
with open("local_data/prep_out.json", encoding="utf-8") as f:
	PREPOUT = json.load(f)
MEANS = PREPOUT["means"]
STDS = PREPOUT["stds"]
PATHS = ("local_data/aerial_prepared.npz", "local_data/target_prepared.npz")

def _load(paths):

	"""
	Loads numpy arrays with aerial and target values
	"""

	aerial = np.load(paths[0])["arr_0"]
	target = np.load(paths[1])["arr_0"]

	return aerial, target

def _reconstruct_aerial(standardized: np.ndarray):

	if len(standardized.shape) == 3:
		standardized = standardized.reshape((1, *standardized.shape))

	void_pixels = (standardized==0).all(axis=3)
	for i in range(standardized.shape[3]):
		channel = standardized[:, :, :, i]
		channel[~void_pixels] = channel[~void_pixels] * STDS[i] + MEANS[i]
	images = standardized.astype(np.uint8)

	return images


def reconstruct_from_files(paths, idcs=None, show=0):

	"""
	Reconstructs images from saved npz's
	"""

	LOG.log("Loading images...")
	aerial, target = _load(paths)
	LOG.log("Done loading images\n")

	LOG.log("Reconstructing aerial images...")
	# Checks if idcs is None or not iterable
	idcs = idcs if idcs is not None else np.arange(len(aerial))
	idcs = (idcs,) if not hasattr(idcs, "__iter__") else idcs
	aerial = _reconstruct_aerial(aerial[idcs])
	LOG.log("Done reconstructing aerial images\n")

	LOG.log("Showing reconstructed images...")
	if show == -1:
		show = len(idcs)
	for i in range(min(len(idcs), show)):
		print(aerial[i].dtype, aerial[i].shape)
		im = Image.fromarray(aerial[i])
		im.show()


if __name__ == "__main__":

	reconstruct_from_files(PATHS, 12, show=2)
