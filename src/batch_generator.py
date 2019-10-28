import numpy as np

import json

import os, sys
os.chdir(sys.path[0])


def load_data(path):
	# I should: Receive path and load the data in using np.load and save it as two matrices
	data = np.load(path + 'aerial_prepared.npz')
	target = np.load(path + 'target_prepared.npz')
	info = json.load(path + 'prep_out.json')
	
	print(data.shape)
	print(target.shape)

	train_idcs, val_idcs, test_idcs = info["train_idcs"], info["val_idcs"], info["test_idcs"]


	train_data, train_target = data[train_idcs], target[train_idcs]
	val_data, val_target = data[val_idcs], target[val_idcs]
	test_data, test_target = data[test_idcs], target[test_idcs]

	return train_data, train_target, val_data, val_target, test_data, test_target

def batch_generator(batch_size, data, target, train = True, classes = 3):
	num_imgs = data.shape[0]
	chosen_idcs = np.random.choice(num_imgs, batch_size, replace = False)
	
	batch_data = data[chosen_idcs]
	batch_target = target[chosen_idcs]
#	if train:
#	

	#Normalize and one-hot-encode target
	batch_target = batch_target // 255
	yellow_value = np.array([0,0,1])
	
	#Yellow = red + green :-) <3 
	yellows = batch_target[(batch_target[:,:,0] == 1) & (batch_target[:,:,1] == 1)]
	batch_data[yellows] = yellow_value
	
	pass

if __name__ == "__main__":
	
	train_data, train_target, val_data, val_target, test_data, test_target = load_data('local_data/')
