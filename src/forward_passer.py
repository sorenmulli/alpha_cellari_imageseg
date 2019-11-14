import os, sys

import json
from PIL import Image
import numpy as np
import torch

from data_loader import DataLoader
from farmors_syning import stitch
from image_reconstructor import ensure_shape, ImageReconstructor
from model import example_architecture, Net

with open("local_data/prep_out.json", encoding="utf-8") as f:
	CFG = json.load(f)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Mere sigende navn
def classify_images(net: torch.nn.Module, idcs: np.ndarray = None, perform_stitch = True, save_paths = None):

	"""
	idcs: Iterable of integers of prepared aerial images
	If None: All images will be used
	perform_stitch: If all images are used, they will be stitched together
	save_paths: string if perform_stitch else iterable or None
	"""

	# Loads data and performs input validation
	x = ensure_shape(DataLoader.load(CFG["aerial_path"]))
	y = np.empty_like(x)
	if idcs is not None:
		x = ensure_shape(x[idcs])
		if save_paths is not None:
			assert isinstance(save_paths, (list, tuple, np.ndarray))
	voids = (x == 0).all(axis=1)
	x = torch.from_numpy(x).float().to(DEVICE)

	# Performs forward pass and finds voids
	with torch.no_grad():
		for i in range(x.shape[0]):
			y[i] = net(ensure_shape(x[i])).cpu().numpy()

	# Performs reconstruction
	recontructor = ImageReconstructor()
	reconst = recontructor.reconstruct_output(y, voids)

	# Stiching and saving
	if idcs == None and perform_stitch:
		reconst = stitch(reconst, CFG["split_shape"], False, savepath = save_paths)
	elif save_paths:
		for i in range(len(reconst)):
			img = Image.fromarray(reconst[i]).save(save_paths[i])

	return reconst
	




if __name__ == "__main__":
	os.chdir(sys.path[0])

	# Test case
	net = Net(example_architecture)
	classify_images(net, save_paths="local_data/test-forward-pass.png")






