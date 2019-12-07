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
from logger import get_timestamp, Logger, NullLogger
from model import Net


from matplotlib import pyplot as plt 
import matplotlib.animation as anim

from utilities import class_weight_counter


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Trainer:
	
	def __init__(self, json_path: str, logger=NullLogger()):

		self.json_path = json_path
		with open(json_path, encoding="utf-8") as f:
			self.cfg = json.load(f)
		self.log = logger

	def model_trainer(self, architecture: dict, learning_rate: float, augmentations: AugmentationConfig, epochs: int,
			batch_size: int, val_every: int = 1, with_plot: bool = True, with_accuracies_print: bool = False, save_every: int = 100):
		
		augmenter = Augmenter(augment_cfg=augmentations)
		data_loader = DataLoader(self.json_path, batch_size, augment = augmenter)

		net = Net(architecture).to(DEVICE)

		try:
			ignore_index = self.cfg["classes"].index("0"*9)
		except ValueError:
			ignore_index = -100
		criterion = nn.CrossEntropyLoss(weight = class_weight_counter(data_loader.train_y), ignore_index=ignore_index)
		optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

		self.log(f"Train size: {len(data_loader.train_x)}\nEval size: {len(data_loader.val_x)}\nTest size: {len(data_loader.get_test()[0])}\n")
		self.log(f"Neural network information\n\t{net}")
		
		full_training_loss = list()
		full_eval_loss = list()
		val_iter = list()
		for epoch_idx in range(epochs):
			if epoch_idx % save_every == 0 and epoch_idx != 0:
				self.log("Saving Network ...")
				net.save(f"local_data/wip_model_epoch{epoch_idx}")

			if epoch_idx % val_every == 0:
				with torch.no_grad():
					net.eval()

					val_data, val_target = data_loader.get_validation()
					val_output = net(val_data)

					evalution_loss = criterion(val_output, val_target)
					
					full_eval_loss.append(float(evalution_loss))
					del val_data
					del val_target
					self.log(f"Epoch {epoch_idx}: Evaluation loss: {float(evalution_loss)}")

					#Overwrite name 
					val_data, val_target = data_loader.train_x, data_loader.train_y
					val_output = net(val_data)

					training_loss = criterion(val_output, val_target)
					full_training_loss.append(float(training_loss))
					del val_data
					del val_target

					self.log(f"Epoch {epoch_idx}: Training loss:   {float(training_loss)}\n")
					
					val_iter.append(epoch_idx)

					torch.cuda.empty_cache()
					
			if with_accuracies_print:
				self.log("Accuracy measures: Global acc.: {G:.4}\nClass acc.: {C:.4}\nMean IoU.: {mIoU:.4}\nBound. F1: {BF:.4}\n".format(**accuracy_measures(val_target, val_output)))

			net.train()
			for batch_data, batch_target in data_loader.generate_epoch():
				#targets = torch.argmax(batch_target, dim = 1, keepdim = True).squeeze()
				output = net(batch_data)
				batch_loss = criterion(output, batch_target)

				optimizer.zero_grad()
				batch_loss.backward()
				optimizer.step()
				torch.cuda.empty_cache()		

		if with_plot:
			plt.figure(figsize=(19.2, 10.8))
			plt.plot(val_iter, full_eval_loss, 'r', label="Validation loss")
			plt.plot(val_iter, full_training_loss, 'b', label="Training loss")
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
		"probs": 0.25,
	}

	learning_rate = 2e-4

	augmentations = AugmentationConfig(
		augments =  [flip_lr, flip_tb],  
		cropsize = (250, 250), 
		augment_p = [0.5, 0.5]
	)

	batch_size = 3
	epochs = 100

	logger = Logger("logs/train_run.log", "Running full training loop")
	trainer = Trainer("local_data/prep_out.json", logger)
	net = trainer.model_trainer(architecture, learning_rate, augmentations, epochs, batch_size, val_every = 10, save_every = 1)
	full_forward(net, None, True, "local_data/full-forward.png")
	
