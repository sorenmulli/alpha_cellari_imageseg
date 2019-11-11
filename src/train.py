import os, sys

from evaluation import global_score

import torch
from torch import nn

import numpy as np

from data_loader import DataLoader
from forward_passer import classify_images
from logger import get_timestamp, Logger
from model import Net

from augment import Augmenter, AugmentationConfig, flip_lr, flip_tb

from matplotlib import pyplot as plt 
import matplotlib.animation as anim

from utilities import class_weight_counter

JSON_PATH = "local_data/prep_out.json"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")

LOG = Logger("logs/training_loop_test.log", "Testing Training Loop")


def model_trainer(architecture: dict, learning_rate: float, augmentations: AugmentationConfig,  epochs: int, batch_size: int, val_every: int = 1, with_plot: bool = True):
	
	augmenter = Augmenter(augment_cfg=augmentations)
	data_loader = DataLoader(JSON_PATH, batch_size, augment = augmenter)

	net = Net(architecture).to(DEVICE)

	criterion = nn.CrossEntropyLoss(weight = class_weight_counter(data_loader.train_y))
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


	LOG(f"Train size: {len(data_loader.train_x)}\n Eval size: {len(data_loader.val_x)}\nTest size: {len(data_loader.get_test()[0])}")
	
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

				output = net(val_data)
				
				global_score(val_target, output)

				evalution_loss = criterion(output, val_target)

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
			LOG(f"Epoch {epoch_idx}: Evaluation loss: {float(evalution_loss)}\n")
		if epoch_idx == epochs-1:
			if with_plot:
				plt.figure(figsize=(19.2, 10.8))
				plt.plot(valid_iter, full_eval_loss, 'r', label="Validation loss")
				plt.plot(train_iter, full_training_loss, 'b', label="Training loss")
				plt.xlabel("Epoch")
				plt.ylabel(str(criterion))
				plt.legend()
				plt.show()				
	return net
if __name__ == "__main__":
	os.chdir(sys.path[0])



	architecture = {
	"kernel_size":  3,
	"padding": 1, 
	"stride": 1,
	"pool_dims": (2, 2),
	"probs": 0.5,}


	learning_rate = 5e-4



	augmentations = AugmentationConfig(
	augments =  [flip_lr, flip_tb],  
	cropsize = (250, 250), 
	augment_p = [0.3, 0.3])

	batch_size = 3
	epochs = 10

	net = model_trainer(architecture, learning_rate, augmentations, epochs, batch_size, val_every = 1)
	classify_images(net, None, True, "train-forward-test.png")
#net.save(f"local_data/models/{get_timestamp(True)}-model.pt")
