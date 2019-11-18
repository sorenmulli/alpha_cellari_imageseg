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
		output = torch.empty_like(test_x)
		with torch.no_grad():
			net.eval()
			for i, x in enumerate(test_x):
				log("Forward passing test image %i" % i)
				output[i] = net(ensure_shape(x)).squeeze()
		self.log("Done performing forward passes\n")
		
		self.log("Calculating accuracy measures...")
		measures = accuracy_measures(test_y, output.argmax(dim=1), is_onehot=False)
		self.log("Accuracy measures: Global acc.: {G:.4}\nClass acc.: {C:.4}\nMean IoU.: {mIoU:.4}\nBound. F1: {BF:.4}\n"
			.format(**measures))
		
		self.log("Reconstructing images...")
		output = output.cpu().numpy()
		voids = (test_x == 0).all(dim=1).cpu().numpy()
		reconstructed = ImageReconstructor(json_path=self.json_path).reconstruct_output(output, voids=voids)
		self.log("Done reconstructing images\n")

		# Creating output image with aerial, target, and reconstructed outputs
		if output_dir:
			self.log("Saving test images and reconstructions to directory %s..." % output_dir)
			os.makedirs(output_dir, exist_ok=True)
			shape = reconstructed[0].shape
			space = 20  # Number of white pixels in between aerial, target, and output images
			for i, ti in enumerate(self.cfg["test_idcs"]):
				out_path = output_dir + "/test-reconst_%i.png" % ti
				aerial_path = self.cfg["sub_imgs_folder"].format(type="aerial", index=ti)
				target_path = self.cfg["sub_imgs_folder"].format(type="target", index=ti)
				aerial = np.asarray(Image.open(aerial_path))
				target = np.asarray(Image.open(target_path))

				# Creating output image initialized as white
				img_arr = np.ones(
					(reconstructed[i].shape[0], reconstructed[i].shape[1] * 3 + space * 2, reconstructed[i].shape[2]),
					dtype=np.uint8
				) * 255
				img_arr[:, :shape[1]] = aerial  # Inserting aerial image
				img_arr[:, shape[1]+space:shape[1]*2+space] = target  # Inserting ground truth
				img_arr[:, shape[1]*2+space*2:] = reconstructed[i]  # Inserting reconstructed output
				img = Image.fromarray(img_arr)
				img.save(out_path)
			self.log("Done saving images and reconstructions\n")

		# return measures, reconstructed

if __name__ == "__main__":

	import sys
	os.chdir(sys.path[0])

	log = Logger("logs/test-tester.log", "Testing tester")
	json_path = "local_data/prep_out.json"
	tester = Tester(json_path, log)
	model = Net.from_model("saved_data/soren_tog_run/model")
	tester.test_model(model, "local_data/test")
