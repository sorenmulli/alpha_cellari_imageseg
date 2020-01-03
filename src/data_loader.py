
import numpy as np
import torch
import json
import os

from logger import Logger, NullLogger


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class DataLoader:

	def __init__(self, json_path: str, batch_size: int, augment: callable=None, logger: Logger=None):
		
		with open(json_path, encoding="utf-8") as f:
			self.cfg = json.load(f)

		self.augment = augment if augment else lambda x, y: (x, y)
		
		self.log = logger if logger else NullLogger()
		
		self.log("Loading data...")
		aerial = torch.from_numpy(self.load(self.cfg["aerial_path"]))
		target = torch.from_numpy(self.load(self.cfg["target_path"]))		
		
		self.train_x = aerial[self.cfg["train_idcs"]].float().to(DEVICE)
		self.train_y = target[self.cfg["train_idcs"]].long().to(DEVICE)

		self.val_x = aerial[self.cfg["val_idcs"]].float().to(DEVICE)
		self.val_y = target[self.cfg["val_idcs"]].long().to(DEVICE)
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
			yield [x.to(DEVICE) for x in self._generate_batch(
				idcs[batch*self.batch_size:(batch+1)*self.batch_size]
			)]
	
	def get_validation(self):

		return self.val_x.to(DEVICE), self.val_y.to(DEVICE)
	
	def get_test(self):

		# Returns test data
		# This is not stored in the class instance, as takes up unnecessary memory

		aerial = torch.from_numpy(self.load(self.cfg["aerial_path"]))[self.cfg["test_idcs"]].float().to(DEVICE)
		target = torch.from_numpy(self.load(self.cfg["target_path"]))[self.cfg["test_idcs"]].long().to(DEVICE)

		return aerial, target

#if __name__ == "__main__":
	# Testing
#	logger = Logger("logs/data_loader_test.log", "Testing DataLoader")
#	data_loader = DataLoader(
#		11,
#		lambda x, y: (x, y),
#		logger
#	)
#	logger("Epoch")
#	for i, (train_x, train_y) in enumerate(data_loader.generate_epoch()):
#		logger(i, train_x.shape, train_y.shape, with_timestamp=False)
#	logger.newline()

#	logger("Validation\n%s\n" % (data_loader.get_validation()[0].cpu().numpy().shape,))

#	logger("Test\n%s\n" % (data_loader.get_test()[0].cpu().numpy().shape,))

