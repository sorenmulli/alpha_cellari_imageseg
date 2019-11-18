import numpy as np
import data_prepper as dp
from logger import Logger 
import os, sys
import torch

from model import Net
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def data_loading():
	LOG = Logger("logs/data_prepper.log", "Testing stiching")

	aerial, target = [dp._load_image(x) for x in dp.IMAGE_PATHS]
	aerial, means, stds = dp._standardize(aerial)

	target = dp._create_one_hot(target)
	target = dp._target_index(target)

	aerial, target = dp._pad(aerial, dp.IMAGE_SHAPE[2], mirror_padding = False), dp._pad(target, None, mirror_padding = False)

	aerial, split_shape = dp._split_image(aerial, dp.IMAGE_SHAPE[2])
	target, _ = dp._split_image(target, None)

	aerial = np.transpose(aerial, (0, 3, 1, 2))

	return aerial, target

if __name__ == "__main__":
	os.chdir(sys.path[0])
	net = Net.from_model('saved_data/soren_tog_run/model')
	aerial, target = data_loading()

	
