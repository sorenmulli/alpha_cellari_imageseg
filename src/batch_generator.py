import numpy as np

import json

import os, sys
os.chdir(sys.path[0])

from logger import Logger
from augmentations import data_augment
LOG = Logger("logs/batch_generator.log", "Loading and generating batches from data." )
DATA_PATH = 'local_data/'



def load_data(path):
	# I should: Receive path and load the data in using np.load and save it as two matrices
	with np.load(path + 'aerial_prepared.npz') as datafile:
		data = datafile.f.arr_0
	
	with np.load(path + 'target_prepared.npz') as targetfile:
		target = targetfile.f.arr_0
	
	with open(path + 'prep_out.json') as infofile:
		info = json.load(infofile)
	
	train_idcs, val_idcs, test_idcs = info["train_idcs"], info["val_idcs"], info["test_idcs"]


	train_data, train_target = data[train_idcs], target[train_idcs]
	val_data, val_target = data[val_idcs], target[val_idcs]
	test_data, test_target = data[test_idcs], target[test_idcs]

	return train_data, train_target, val_data, val_target, test_data, test_target


def batch_generator(batch_size, data, target, train = True):
	num_imgs = data.shape[0]
	chosen_idcs = np.random.choice(num_imgs, batch_size, replace = False)
	
	batch_data = data[chosen_idcs]
	batch_target = target[chosen_idcs]
	
	if train:
		batch_data, batch_target = data_augment(batch_data, batch_target)
	
	return batch_data, batch_target

if __name__ == "__main__":
	LOG.log(f"Loading data from {DATA_PATH}.")
	train_data, train_target, val_data, val_target, test_data, test_target = load_data(DATA_PATH)
	LOG.log(f"Data loaded. Dimensions are (data and target): \
	\n\tTrain: {train_data.shape} and {train_target.shape}   \
	\n\tValidation: {val_data.shape} and {val_target.shape}  \
	\n\tTest: {test_data.shape} and {test_target.shape} ")
	
	test_batch_size = 10
	LOG.log(f"Attempting to generate batch of size {test_batch_size}")
	batch_generator(test_batch_size, train_data, train_target)