"""
Module that contains that some utility functions
"""

import os
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset

class ActionDataset(Dataset):
	def __init__(self, data_dir, crop=None, subset = None):
		self.data_dir = data_dir
		self.crop = crop
		
		if subset is None:
			self.trajs = os.listdir(data_dir)
		else:
			self.trajs = [fn for fn in os.listdir(data_dir) if fn in subset]
		
		self._cache = {}
		
	
	def __len__(self):
		return len(self.trajs)//3
		
	def __getitem__(self, idx):
		if idx not in self._cache:
			imgs = np.load(os.path.join(self.data_dir, '%04d_img.npy'%idx))
			actions = np.load(os.path.join(self.data_dir, '%04d_action.npy'%idx)).astype(np.uint8)
			states = np.load(os.path.join(self.data_dir, '%04d_state.npy'%idx))
			
			self._cache[idx] = (imgs, states, actions)
		
		imgs, states, actions = self._cache[idx]
		
		if self.crop is not None:
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