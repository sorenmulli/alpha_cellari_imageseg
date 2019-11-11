import torch 
from utilities import softmax_output_to_prediction, class_weight_counter

import numpy as np

from sklearn.metrics import jaccard_similarity_score 
from sklearn.metrics import confusion_matrix

def accuracy_measures(y_true_tensor: torch.Tensor, network_output: torch.Tensor, measures: dict = {'G': True, 'C': True, 'mIoU': True, 'BF': True}):
	### Assumes that it receives images of shape: (# images, height, width)
	y_pred_tensor = softmax_output_to_prediction(network_output)
	y_true, y_pred = y_true_tensor.flatten().detach().numpy(), y_pred_tensor.flatten().detach().numpy()

	n_classes =  network_output.size()[1]
	conf_matrix = confusion_matrix(y_true, y_pred)
	output = dict()

	if measures['G']:
		output['G'] = conf_matrix.trace() / conf_matrix.sum() 
	if measures['C']:
		class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis = 1)
		output['C'] = class_accuracies.mean()
	if measures['mIoU']:
		IoUs = np.empty(3)
		for class_ in  range(n_classes):
			IoUs[class_] =  conf_matrix[class_, class_] / (conf_matrix[class_].sum()  + conf_matrix[:, class_].sum() -  conf_matrix[class_, class_] )
		output['mIoU'] = IoUs.mean()
	if measures['BF']:
		F1s = np.empty(3)
		for class_ in  range(n_classes):
			F1s[class_] = 2* conf_matrix[class_, class_] / (2*conf_matrix[class_].sum()  + conf_matrix[:, class_].sum() - 2* conf_matrix[class_, class_] )
		output['BF'] = F1s.mean()
	
	return output