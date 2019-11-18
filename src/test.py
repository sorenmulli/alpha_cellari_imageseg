import os

import json
import numpy as np
import torch
from PIL import Image

from data_loader import DataLoader
from evaluation import accuracy_measures
from image_reconstructor import ensure_shape, ImageReconstructor
from logger import Logger, NullLogger
from model import example_architecture, Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tester:

	def __init__(self, json_path: str, logger: Logger = NullLogger()):

		self.json_path = json_path
		with open(json_path, encoding="utf-8") as f:
			self.cfg = json.load(f)
		self.log = logger

	def test_model(self, net: Net, output_dir: str = ""):

		"""
		Calculates accuracy scores on a net using test images
		"""

		self.log("Loading test data...")
		test_x = torch.from_numpy(DataLoader.load(self.cfg["aerial_path"])[self.cfg["test_idcs"]]).to(DEVICE).float()
		test_y = torch.from_numpy(DataLoader.load(self.cfg["target_path"])[self.cfg["test_idcs"]]).to(DEVICE).long()
		self.log("Done loading test data\n")

		self.log("Performing forward passes...")
		shape = list(test_x.shape)
		shape[0] = 1
		output = torch.empty(shape).to(test_y.dtype).to(test_y.device)
		print(net)
		with torch.no_grad():
			net.eval()
			for i, x in enumerate(test_x):
				print((x==0).sum())
				output[0] = net(ensure_shape(x)).squeeze()
				break

		self.log("Done performing forward passes\n")
		# measures = accuracy_measures(test_y, output.argmax(dim=1), is_onehot=False)
		# self.log("Accuracy measures: Global acc.: {G:.4}\nClass acc.: {C:.4}\nMean IoU.: {mIoU:.4}\nBound. F1: {BF:.4}\n"
		# 	.format(**measures))
		
		self.log("Reconstructing images...")
		output = output.cpu().numpy()
		voids = (ensure_shape(test_x[0] == 0)).all(dim=1).cpu().numpy()
		reconstructed = ImageReconstructor(json_path=self.json_path).reconstruct_output(output, voids=voids)
		self.log("Done reconstructing images\n")

		if output_dir:
			self.log("Saving test images and reconstructions to directory %s..." % output_dir)
			if not os.path.exists(output_dir):
				os.makedirs(output_dir)
			for i in range(len(reconstructed)):
				path = output_dir + "/test-reconst_%i.png" % i
				space = 20
				img_arr = np.zeros(
					(reconstructed[i].shape[0], reconstructed[i].shape[1] * 2 + space, reconstructed[i].shape[2]),
					dtype=np.uint8
				)
				img_arr[:, reconstructed[i].shape[1]+space:] = reconstructed[i]
				img = Image.fromarray(img_arr)
				img.save(path)
			self.log("Done saving images and reconstructions\n")

		# return measures, reconstructed

if __name__ == "__main__":

	import sys
	os.chdir(sys.path[0])

	log = Logger("logs/test-tester.log", "Testing tester")
	json_path = "local_data/prep_out.json"
	tester = Tester(json_path, log)
	model = Net.from_model("saved_data/soren_tog_run/model", log)
	tester.test_model(model, "local_data/test")

