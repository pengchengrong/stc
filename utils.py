"""
Module that contains that some utility functions
"""

import os
import numpy as np
import torch
import glob

from torch.utils.data import DataLoader, Dataset

state_mean = [2.5204e+00,  1.4577e+01,  0,  1.4897e+00, 0, 0]
state_std = [1.4174, 3.3497, 1, 0.8749, 0.7345, 2.8847]
img_mean = [0.2412, 0.2214, 0.2262]
img_std = [0.2095, 0.2114, 0.2541]

def pre_process_img(imgs):
	imgs = imgs / 255.
	imgs[:, :, :, 0] = (imgs[:, :, :, 0] - img_mean[0]) / img_std[0]
	imgs[:, :, :, 1] = (imgs[:, :, :, 1] - img_mean[1]) / img_std[1]
	imgs[:, :, :, 2] = (imgs[:, :, :, 2] - img_mean[2]) / img_std[2]
	return imgs

def pre_process_state(state):
	#statistics of the state array
	#std= tensor([3.3618, 1.4174, 3.3497, 0.0138, 0.8749, 0.7345, 2.8847, 0.9911])
	#mean= tensor([ 1.4567e+01,  2.5204e+00,  1.4577e+01,  7.4328e-04,  1.4897e+00,-2.8953e-02, -5.4145e-02,  1.8194e+00])
	state = state[:, 1:7]
	state = (state - state_mean) / state_std
	return state

class ActionDataset(Dataset):

	def __init__(self, data_dir, crop=None, subset = None):
		self.data_dir = data_dir
		self.crop = crop
		
		self.trajs = sorted(glob.glob(data_dir + "/*_img.npy"), key=os.path.getmtime) #os.listdir(data_dir)
		if subset is not None:
			self.trajs = self.trajs[-1*subset:]
		
		self._cache = {}
		
	
	def __len__(self):
		return len(self.trajs)
		
	def __getitem__(self, idx):

		img_file = self.trajs[idx]
		action_file = img_file.replace("_img.", "_action.")
		state_file = img_file.replace("_img.", "_state.")
		#imgs = np.load(img_file).astype(np.uint8)
		imgs = np.load(img_file)
		actions = np.load(action_file).astype(np.uint8)
		states = np.load(state_file)
		#imgs = pre_process_img(imgs)
		states = pre_process_state(states)			
				
		if self.crop is not None and len(imgs) > self.crop:
			s = np.random.choice(len(imgs) - self.crop + 1)
			imgs = imgs[s:s+self.crop]
			states = states[s:s+self.crop]
			actions = actions[s:s+self.crop]

		#imgs = (imgs - [74.26,69.21,61.61]) / [5.58,5.24,4.83]

		return imgs, states, actions



		if idx not in self._cache:
			img_file = self.trajs[idx]
			action_file = img_file.replace("_img.", "_action.")
			state_file = img_file.replace("_img.", "_state.")
			#imgs = np.load(img_file).astype(np.uint8)
			imgs = np.load(img_file)
			actions = np.load(action_file).astype(np.uint8)
			states = np.load(state_file)
			#imgs = pre_process_img(imgs)
			states = pre_process_state(states)
			
			self._cache[idx] = (imgs, states, actions)
		
		imgs, states, actions = self._cache[idx]
		
		if self.crop is not None and len(imgs) > self.crop:
			s = np.random.choice(len(imgs) - self.crop + 1)
			imgs = imgs[s:s+self.crop]
			states = states[s:s+self.crop]
			actions = actions[s:s+self.crop]

		#imgs = (imgs - [74.26,69.21,61.61]) / [5.58,5.24,4.83]

		return imgs, states, actions
		
		
def load(data_filepath, num_workers=0, batch_size=32, **kwargs):
	dataset = ActionDataset(data_filepath, **kwargs)
	return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

class SummaryWriter:
	def __init__(self, *args, **kwargs):
		print("tensorboardX not found. You need to install it to use the SummaryWriter.")
		print("try: pip3 install tensorboardX")
		raise ImportError
try:
	from tensorboardX import SummaryWriter
except ImportError:
	pass