import numpy as np
#Operation, chance of applying operation
def _mirror(image): 

	return image

def _flip(image):
	return image

AUGMENTATION_DICT = {
	"crop_size": (450, 450),
	"augmentations":
	{
	_mirror: 0.1,
	_flip : 0.1
	}
}




def data_augment(data, target):
	##########################
	#IMPLEMENT CROPPING FIRST#
	##########################
	##########################
	for i, image in enumerate(data):
		for augmentation, chance in AUGMENTATION_DICT["augmentations"].items():
			if np.random.rand() < chance:
				image = augmentation(image)
				target[i] = augmentation(image)


	return data, target