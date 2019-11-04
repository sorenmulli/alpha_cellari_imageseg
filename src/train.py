import os, sys
os.chdir(sys.path[0])


import torch
from torch import nn

import numpy as np

from data_loader import DataLoader
from logger import get_timestamp, Logger
from model import Net

from time import sleep


ARCHITECTURE = {
	"kernel_size":  3,
	"padding": 1, 
	"stride": 1,
	"pool_dims": (2, 2), 
}
LEARNING_RATE = 5e-4


BATCH_SIZE = 7
EPOCHS = 10
VAL_EVERY = 1

JSON_PATH = "local_data/prep_out.json"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")

######################
# Initialization
LOG = Logger("logs/training_loop_test.log", "Testing Training Loop")
data_loader = DataLoader(
	JSON_PATH,
	BATCH_SIZE,
#	logger = LOG
)
net = Net(ARCHITECTURE).to(DEVICE)

####################

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

#####################

LOG(f"""
Train size: {len(data_loader.train_x)}
Eval size: {len(data_loader.val_x)}
Test size: {len(data_loader.get_test()[0])} 
""")

#####################
for epoch_idx in range(EPOCHS):
	if not epoch_idx % VAL_EVERY:
		net.eval()

		val_data, val_target = data_loader.get_validation() 
		
		#targets = torch.argmax(val_target, dim = 1, keepdim = True).squeeze()

		output = net(val_data)
		
		evalution_loss = criterion(output, val_target)
		LOG(f"Evaluation loss: {float(evalution_loss)}")
		 
	net.train()

	training_loss = list()
	for batch_data, batch_target in data_loader.generate_epoch():
		#targets = torch.argmax(batch_target, dim = 1, keepdim = True).squeeze()
		output = net(batch_data)
		batch_loss = criterion(output, batch_target)

		optimizer.zero_grad()
		batch_loss.backward()
		optimizer.step()

		training_loss.append(float(batch_loss))
		
	LOG(f"Training loss: {np.mean(training_loss)}")


		

#net.save(f"local_data/models/{get_timestamp(True)}-model.pt")
