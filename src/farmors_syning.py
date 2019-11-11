import os, sys
os.chdir(sys.path[0])

import numpy as np
import torch

from logger import Logger, NullLogger

class Stitcher:

	# Stitcher et 154 x channels x height x width array sammen til height x width x channels

	def __init__(self, logger: Logger=NullLogger):

		self.log = logger
	
	def stitch(self, images: np.ndarray or torch.tensor):

		# Checks if 
		if isinstance(images, torch.Tensor):
			images = images.cpu().numpy()







