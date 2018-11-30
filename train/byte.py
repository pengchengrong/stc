"""
Module that contains that some utility functions
"""

import os
import numpy as np
import torch
from torch import nn, optim

import glob
imgs = glob.glob("./*_img.npy")
print("About to byte ", len(imgs) , " files.")
for img in imgs:
	img_data = np.load(img)
	img_data = torch.tensor(img_data).byte()
	np.save(img, img_data.numpy())
	print(img, " is byted.")
