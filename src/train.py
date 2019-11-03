import torch

from data_loader import DataLoader
from logger import get_timestamp, Logger
from model import Net

from time import sleep

JSON_PATH = "local_data/prep_out.json"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# DEVICE = torch.device("cpu")

ARCHITECTURE = {
	"kernel_size":  3,
	"padding": 1, 
	"stride": 1,
	"pool_dims": (2, 2), 
}
BATCH_SIZE = 7

######################
LOG = Logger("logs/training_loop_test.log", "Testing Training Loop")
data_loader = DataLoader(
	BATCH_SIZE,
	logger = LOG
)
net = Net(ARCHITECTURE, LOG).to(DEVICE)
#####################

for batch_data, batch_target in data_loader.generate_epoch():
	#I SHOULD NOT BE NECESSARY: GET THIS INTO data_loader
	LOG("Testing forward pass")
	net(batch_data)
	LOG("Forward pass completed\n")

net.save(f"local_data/models/{get_timestamp(True)}-model.pt")




