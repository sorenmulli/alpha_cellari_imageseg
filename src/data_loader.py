import os, sys
os.chdir(sys.path[0])

import numpy as np
import torch
import json

from logger import Logger
from image_reconstructor import load_npz
from augmentations import data_augment

JSON_PATH = "local_data/prep_out.json"
CPU = torch.device("cpu")
GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DataLoader:

	def __init__(self, batch_size: int, augment: callable, logger: Logger=None):
		
		self.augment = augment
		if logger:
			self.log = logger
		else:
			self.log = lambda x: 0

		with open(JSON_PATH, encoding="utf-8") as f:
			self.cfg = json.load(f)
		
		self.log("Loading data...")
		self.aerial = torch.from_numpy(load_npz(self.cfg["aerial_path"])).to(GPU)
		self.target = torch.from_numpy(load_npz(self.cfg["target_path"])).to(GPU)

		self.train_x = self.aerial[self.cfg["train_idcs"]]
		self.train_y = self.target[self.cfg["train_idcs"]]
		self.val_x = self.aerial[self.cfg["val_idcs"]]
		self.val_y = self.target[self.cfg["val_idcs"]]
		self.test_x = self.aerial[self.cfg["test_idcs"]]
		self.test_y = self.target[self.cfg["test_idcs"]]
		self.log("Done loading %i images\n" % len(self.aerial))
	
		self.batch_size = batch_size
		self.n_batches = len(self.train_x) // batch_size

	def _generate_batch(self, idcs: np.ndarray):

		return self.augment(self.train_x[idcs], self.train_y[idcs])
	
	def generate_epoch(self):

		self.log("Generating epoch...")
		idcs = np.arange(len(self.train_x))
		np.random.shuffle(idcs)
		for batch in range(self.n_batches):
			self.log("Yielding batch %i" % batch)
			yield self._generate_batch(
				idcs[batch*self.batch_size:(batch+1)*self.batch_size]
			)
	
	def get_validation(self):

		return self.augment(self.val_x, self.val_y)
	
	def get_test(self):

		return self.augment(self.test_x, self.test_y)

if __name__ == "__main__":
	data_loader = DataLoader(
		11,
		lambda x, y: (x, y),
		Logger("logs/data_loader_test.log", "Testing DataLoader")
	)
	for i, (train, test) in enumerate(data_loader.generate_epoch()):
		print(i, train.shape, test.shape)
	print(data_loader.get_validation()[0].shape)
	print(data_loader.get_test()[0].shape)

