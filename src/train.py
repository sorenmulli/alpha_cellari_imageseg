import torch

from data_loader import DataLoader
from logger import Logger
from model import Net

JSON_PATH = "local_data/prep_out.json"
CPU = torch.device("cpu")
GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARCHITECTURE = {
	"kernel_size":  3,
	"padding": 1, 
	"stride": 1,
	"pool_dims": (2, 2), 
}


######################
logger = Logger("logs/training_loop_test.log", "Testing Training Loop")
data_loader = DataLoader(
	11,
	logger = logger
)
net = Net(ARCHITECTURE)
#####################

for batch_data, batch_target in data_loader.generate_epoch():
	net(batch_data)