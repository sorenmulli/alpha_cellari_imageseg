import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import random
import os


def batch_generator(img_list, anno_list, num_of_classes=2):
    '''Example batch generator, Please note this example
    does not work for more classes than 2. You have to expand the code
    in order to handle that (i.e, background, weed and crops)
    '''

    #Get random image
    rand_int = random.randint(0, len(img_list))
    img = np.array(Image.open(img_list[rand_int]))
    anno = np.array(Image.open(anno_list[rand_int]).convert("L"))

    anno[anno>0] = 1
    anno_reshaped = np.zeros((anno.shape[0], anno.shape[1], num_of_classes))

    #IMAGE AUGMENTATION SHOULD HAPPEN HERE.
    #EXAMPLES : RANDOM CROPPING, RANDOM FLIPPING
    #MIRRORING, ZOOMING, MEAN SUBTRACTION AND STD DIVISION.

    for i in range(num_of_classes):
        anno_reshaped[:, :, i][anno == i] = 1

    return img, anno_reshaped

#Gland images are bmp, my corrosponding annotations are converted pngs.

path_to_images = '/home/peter/Desktop/datasets/Gland/gland/train/raw_images'
path_to_annotations = '/home/peter/Desktop/datasets/Gland/gland/train/annotations_white'

img_list = glob.glob(path_to_images+os.sep+"*.bmp")
img_list = sorted(img_list)
anno_list = glob.glob(path_to_annotations+os.sep+"*.png")
anno_list = sorted(anno_list)

img, anno = batch_generator(img_list, anno_list,2)

#Please note, if done correctly, you should not be able to view your raw image directly
#with matplotlib due to the subtraction of the mean, and pil not understanding floats.

plt.imshow(img)
plt.show()
plt.imshow(anno[:, :, 0])
plt.show()
plt.imshow(anno[:, :, 1])
plt.show()