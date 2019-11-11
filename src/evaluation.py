import torch 
from utilities import softmax_output_to_prediction
 

def global_score(y_true_tensor: torch.Tensor, network_output: torch.Tensor):
	### Assumes that it receives images of shape: (# images, height, width)
	y_pred_tensor = softmax_output_to_prediction(network_output)
	
	y_true, y_pred = y_true_tensor.flatten(), y_pred_tensor.flatten()

	return torch.mean((y_true == y_pred).float()) 
