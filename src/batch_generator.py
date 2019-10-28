import numpy as np

import json

import os, sys
os.chdir(sys.path[0])


def load_data(path):
	# I should: Receive path and load the data in using np.load and save it as two matrices
	data = np.load(path + 'aerial_prepared.npz')
	target = np.load(path + 'target_prepared.npz')
	info = json.load(path + 'prep_out.json')
	
	train_idcs, val_idcs, test_idcs = info["train_idcs"], info["val_idcs"], info["test_idcs"]


	train_data, train_target, val_data, val_test, test_data, test_target = 'foo', 'foo', 'foo', 'foo', 'foo', 'foo'

	return train_data, train_target, val_data, val_test, test_data, test_target


if __name__ == "__main__":
	
	train_data, train_target, val_data, val_test, test_data, test_target = load_data('local_data/')
