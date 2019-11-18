import os, sys


import json
import numpy as np
import torch
from torch import nn

from augment import Augmenter, AugmentationConfig, flip_lr, flip_tb
from data_loader import DataLoader
from evaluation import accuracy_measures
from forward_passer import full_forward
from image_reconstructor import ensure_shape
from logger import get_timestamp, Logger
from model import Net
from test import Tester


from matplotlib import pyplot as plt 
import matplotlib.animation as anim

from utilities import class_weight_counter

JSON_PATH = "local_data/prep_out.json"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#DEVICE = torch.device("cpu")

LOG = Logger("logs/training_loop_test.log", "Testing Training Loop")


def model_trainer(architecture: dict, learning_rate: float, augmentations: AugmentationConfig,  epochs: int, batch_size: int, val_every: int = 1, with_plot: bool = True):
	
	augmenter = Augmenter(augment_cfg=augmentations)
	data_loader = DataLoader(JSON_PATH, batch_size, augment = augmenter)

	net = Net(architecture).to(DEVICE)

	criterion = nn.CrossEntropyLoss(weight = class_weight_counter(data_loader.train_y), ignore_index=-1)
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


	LOG(f"Train size: {len(data_loader.train_x)}\nEval size: {len(data_loader.val_x)}\nTest size: {len(data_loader.get_test()[0])}\n")
	LOG(f"Neural network information\n\t{net}")
	full_training_loss = list()
	full_eval_loss = list()
	train_iter = list()
	valid_iter = list()

	
	for epoch_idx in range(epochs):
		if epoch_idx % val_every == 0:
			with torch.no_grad():
				net.eval()

				val_data, val_target = data_loader.get_validation() 
				
				#targets = torch.argmax(val_target, dim = 1, keepdim = True).squeeze()

				val_output = net(val_data)
				
				

				evalution_loss = criterion(val_output, val_target)

				
				full_eval_loss.append(float(evalution_loss))
				valid_iter.append(epoch_idx)
			
		net.train()

		training_loss = list()
		for batch_data, batch_target in data_loader.generate_epoch():
			#targets = torch.argmax(batch_target, dim = 1, keepdim = True).squeeze()
			output = net(batch_data)
			batch_loss = criterion(output, batch_target)

			optimizer.zero_grad()
			batch_loss.backward()
			optimizer.step()
			torch.cuda.empty_cache()

			training_loss.append(float(batch_loss))
		train_iter.append(epoch_idx)
		full_training_loss.append(np.mean(training_loss))
		
		if epoch_idx % val_every == 0:
			LOG(f"Epoch {epoch_idx}: Training loss:   {np.mean(training_loss)}")
			LOG(f"Epoch {epoch_idx}: Evaluation loss: {float(evalution_loss)}")
			LOG("Accuracy measures: Global acc.: {G:.4}\nClass acc.: {C:.4}\nMean IoU.: {mIoU:.4}\nBound. F1: {BF:.4}\n".format(**accuracy_measures(val_target, val_output)))

		if epoch_idx == epochs-1:
			if with_plot:
				plt.figure(figsize=(19.2, 10.8))
				plt.plot(valid_iter, full_eval_loss, 'r', label="Validation loss")
				plt.plot(train_iter, full_training_loss, 'b', label="Training loss")
				plt.xlabel("Epoch")
				plt.ylabel(str(criterion))
				plt.legend()
				plt.show()

	return net, criterion



if __name__ == "__main__":
	os.chdir(sys.path[0])

	architecture = {
		"kernel_size":  3,
		"padding": 1, 
		"stride": 1,
		"pool_dims": (2, 2),
		"probs": 0.1,
	}

	learning_rate = 2e-4

	augmentations = AugmentationConfig(
		augments =  [flip_lr, flip_tb],  
		cropsize = (250, 250), 
		augment_p = [0.5, 0.5]
	)

	batch_size = 3
	epochs = 0

	net, criterion = model_trainer(architecture, learning_rate, augmentations, epochs, batch_size, val_every = 10)
	full_forward(net, None, True, "local_data/full-forward.png")
	# tester = Tester(JSON_PATH, LOG)
	# tester.test_model(net, criterion, "local_data/test")
#net.save(f"local_data/models/{get_timestamp(True)}-model.pt")
