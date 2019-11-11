import torch

def class_weight_counter(y: torch.Tensor):
	_, counts = torch.unique(y, return_counts = True)
	partitions = counts.float() / torch.sum(counts)
	return 1 - partitions

def softmax_output_to_prediction(output: torch.Tensor):
	### Assumes that it receives images of shape: (image #, class #,  height, width)
	return torch.argmax(output, dim = 1, keepdim = True).squeeze()