import torch

def class_weight_counter(y):
	_, counts = torch.unique(y, return_counts = True)
	partitions = counts.float() / torch.sum(counts)
	return 1 - partitions