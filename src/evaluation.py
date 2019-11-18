import torch 
from utilities import softmax_output_to_prediction, class_weight_counter

import numpy as np

from sklearn.metrics import jaccard_similarity_score 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

def accuracy_measures(
		y_true_tensor: torch.Tensor,
		y_pred_tensor: torch.Tensor,
		n_classes: int = 3,
		is_onehot: bool = True,
		measures: dict = {'G': True, 'C': True, 'mIoU': True, 'BF': True}
	):
	### Assumes that it receives images of shape: (# images, height, width)
	
	if is_onehot:
		y_pred_tensor = softmax_output_to_prediction(y_pred_tensor)
	
	y_true, y_pred = y_true_tensor.flatten().detach().cpu().numpy(), y_pred_tensor.flatten().detach().cpu().numpy()
	voids = y_true == -1
	y_true, y_pred = y_true[~voids], y_pred[~voids]
	
	conf_matrix = confusion_matrix(y_true, y_pred)
	output = dict()

	if measures['G']:
		output['G'] = conf_matrix.trace() / conf_matrix.sum() 
	if measures['C']:
		class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis = 1)
		output['C'] = class_accuracies.mean()
	if measures['mIoU']:
		IoUs = np.empty(n_classes)
		for class_ in  range(n_classes):
			IoUs[class_] =  conf_matrix[class_, class_] / (conf_matrix[class_].sum()  + conf_matrix[:, class_].sum() -  conf_matrix[class_, class_] )
		output['mIoU'] = IoUs.mean()
	if measures['BF']:
		F1s = np.empty(n_classes)
		for class_ in  range(n_classes):
			F1s[class_] = 2* conf_matrix[class_, class_] / (conf_matrix[class_].sum()  + conf_matrix[:, class_].sum())	
		output['BF'] = F1s.mean()
	return output