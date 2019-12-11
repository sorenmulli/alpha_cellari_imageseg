import os, sys
os.chdir(sys.path[0])

import numpy as np
from PIL import Image

from data_prepper import _load_image, _pad

def _fix_image():

	img_path = "saved_data/soren_hpc_standard_run/full-forward.png"
	inp, reconst = _load_image("local_data/raw.png"), _load_image(img_path, dtype=np.uint8)
	inp = _pad(inp)
	voids = inp.sum(axis=2) == 0
	void_colour = np.array([255, 255, 255], dtype=np.uint8)
	reconst2 = reconst.copy()
	reconst2[voids] = void_colour
	img = Image.fromarray(reconst2)
	img.save(img_path)


if __name__ == "__main__":
	_fix_image()
