import torch
from data_loader import DataLoader
from logger import Logger

import evaluation

def class_weight_counter(y: torch.Tensor):
	_, counts = torch.unique(y, return_counts = True)
	partitions = counts.float() / torch.sum(counts)
	return 1 - partitions

def softmax_output_to_prediction(output: torch.Tensor):
	### Assumes that it receives images of shape: (image #, class #,  height, width)
	return torch.argmax(output, dim = 1, keepdim = True).squeeze()

def baseline_computation(json_path, log, nclasses = 3):

	data_loader = DataLoader(json_path, 12)
	criterion = torch.nn.CrossEntropyLoss(weight = class_weight_counter(data_loader.train_y))

	baseline_pred = int(torch.argmax(torch.unique(data_loader.train_y, return_counts=True)[1]))
	
	val_target = data_loader.val_y
	test_target = data_loader.get_test()[1]

	val_output = torch.zeros((val_target.shape[0], nclasses, val_target.shape[1], val_target.shape[2]))
	test_output = torch.zeros((test_target.shape[0], nclasses, test_target.shape[1], test_target.shape[2]))

	val_output[:, baseline_pred, :, :] = 1
	test_output[:, baseline_pred, :, :] = 1

	log("Validation accuracy measures: Global acc.: {G:.4}\nClass acc.: {C:.4}\nMean IoU.: {mIoU:.4}\nBound. F1: {BF:.4}\n".format(**evaluation.accuracy_measures(val_target, val_output)))
	log("Test accuracy measures: Global acc.: {G:.4}\nClass acc.: {C:.4}\nMean IoU.: {mIoU:.4}\nBound. F1: {BF:.4}\n".format(**evaluation.accuracy_measures(test_target, test_output)))

	val_loss = criterion(val_output, val_target)
	test_loss = criterion(test_output, test_target)

	log(f"Validation loss: {val_loss:.4}")
	log(f"Test loss: {test_loss:.4}")
if __name__ == "__main__":
	import sys, os
	os.chdir(sys.path[0])
	JSON_PATH = "local_data/prep_out.json"
	LOG = Logger("logs/baseline_test.log", "Evaluating Baseline")

	baseline_computation(JSON_PATH, LOG)