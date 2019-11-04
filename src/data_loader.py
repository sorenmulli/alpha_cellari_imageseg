import os, sys
os.chdir(sys.path[0])

import numpy as np
import torch
import json

from logger import Logger, NullLogger

JSON_PATH = "local_data/prep_out.json"
with open(JSON_PATH, encoding="utf-8") as f:
	CFG = json.load(f)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")

class DataLoader:

	def __init__(self, batch_size: int, augment: callable=None, logger: Logger=None):
		
		self.augment = augment if augment else lambda x, y: (x, y)
		
		self.log = logger if logger else NullLogger()
		
		self.log("Loading data...")
		aerial = torch.from_numpy(self.load(CFG["aerial_path"]))
		target = torch.from_numpy(self.load(CFG["target_path"]))

		self.train_x = aerial[CFG["train_idcs"]].float().to(DEVICE)
		self.train_y = target[CFG["train_idcs"]].bool().to(DEVICE)
		self.val_x = aerial[CFG["val_idcs"]].float().to(DEVICE)
		self.val_y = target[CFG["val_idcs"]].bool().to(DEVICE)
		self.log("Done loading %i images\n" % len(aerial))
	
		self.batch_size = batch_size
		self.n_batches = len(self.train_x) // batch_size
	
	@staticmethod
	def load(path: str):

		"""
		Loads a numpy array saved as a .npy or .npz file
		path should not contain file extension
		"""

		# Prioritizes .npy, as they load much faster
		if os.path.isfile(path+".npy"):
			arr = np.load(path+".npy")
		else:
			arr = np.load(path+".npz")["arr_0"]

		return arr

	def _generate_batch(self, idcs: np.ndarray):

		return self.augment(self.train_x[idcs], self.train_y[idcs])
	
	def generate_epoch(self):

		self.log("Generating epoch...")
		idcs = np.arange(len(self.train_x))
		np.random.shuffle(idcs)
		for batch in range(self.n_batches):
			yield self._generate_batch(
				idcs[batch*self.batch_size:(batch+1)*self.batch_size]
			)
	
	def get_validation(self):

		return self.augment(self.val_x, self.val_y)
	
	def get_test(self):

		# Returns test data
		# This is not stored in the class instance, as takes up unnecessary memory

		aerial = torch.from_numpy(self.load(CFG["aerial_path"]))[CFG["test_idcs"]].float().to(DEVICE)
		target = torch.from_numpy(self.load(CFG["target_path"]))[CFG["test_idcs"]].bool().to(DEVICE)

		return self.augment(aerial, target)

if __name__ == "__main__":
	# Testing
	logger = Logger("logs/data_loader_test.log", "Testing DataLoader")
	data_loader = DataLoader(
		11,
		lambda x, y: (x, y),
		logger
	)
	logger("Epoch")
	for i, (train_x, train_y) in enumerate(data_loader.generate_epoch()):
		logger(i, train_x.shape, train_y.shape, with_timestamp=False)
	logger.newline()

	logger("Validation\n%s\n" % (data_loader.get_validation()[0].cpu().numpy().shape,))

	logger("Test\n%s\n" % (data_loader.get_test()[0].cpu().numpy().shape,))

