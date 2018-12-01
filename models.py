from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from utils import *

class ResBlock(nn.Module):
	def __init__(self, in_channel, out_channel, stride, k=5, pad=2):
		super(ResBlock, self).__init__()
		'''
		Your code here
		'''
		if in_channel == out_channel:
			self.project_input = False
		else:
			self.project_input = True
			self.projection = nn.Conv2d(in_channel, out_channel, stride=stride, kernel_size=1)

		self.bn1 = nn.BatchNorm2d(out_channel)
		self.bn2 = nn.BatchNorm2d(out_channel)
		self.relu = nn.LeakyReLU()
		self.stride = stride

		self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=k, stride=stride, padding=pad)
		self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=k, stride=1, padding=pad)
    
	def forward(self, x):
		'''
		Your code here
		'''
		if self.project_input == True:
			residual = self.projection(x)
		else:
			residual = x

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)

		#print(x.size(), ", ", residual.size())
		x += residual
		x = self.relu(x)
		return x

class Policy:
	def __init__(self, model):
		self.model = model
		self.hist = []
		self.hist_state = []
		
	def __call__(self, obs, state):

		# c, w, h
		'''
		obs[0, :, :] = (obs[0, :, :] - img_mean[0]) / img_std[0]
		obs[1, :, :] = (obs[1, :, :] - img_mean[1]) / img_std[1]
		obs[2, :, :] = (obs[2, :, :] - img_mean[2]) / img_std[2]
		'''

		state = state[1:7]
		state = (state - torch.tensor(state_mean)) / torch.tensor(state_std)

		self.hist.append(obs)
		self.hist_state.append(state)
		if len(self.hist) > 1:
			self.hist = self.hist[-1:]
			self.hist_state = self.hist_state[-1:]
		x = torch.stack(self.hist, dim=0)[None]
		y = torch.stack(self.hist_state, dim=0)[None]
		#print(x.size(), ", ", y.size(), ", ", self.model(x, y), ", ", self.model(x, y)[0,-1,:])
		return self.model(x, y)[0,-1,:]

class Model(nn.Module):

	def __init__(self):
		super().__init__()
		
		'''
		Your code here
		'''
		'''
			nn.Conv2d(3,16,5,3,1),
			nn.ReLU(True),
			nn.Conv2d(16,32,5,3,1),
			nn.ReLU(True),
			nn.Conv2d(32,64,5,3,1),
			nn.ReLU(True),
			nn.Conv2d(64,128,5,3,1),
			nn.ReLU(True),
		'''
		self.conv = nn.Sequential(
			ResBlock(3, 16, 3),
			ResBlock(16, 32, 3),
			ResBlock(32, 64, 3),
			ResBlock(64, 128, 3),
		)

		self.conv_out = 128*4*3 #768
		
		self.fc = nn.Sequential(
			nn.Linear(self.conv_out,58),
			nn.ReLU(True),
		)
		
		self.fc2 = nn.Sequential(
			nn.Linear(64,6),
		)

	def forward(self, hist, state):
		'''
		Your code here
		Input size: (batch_size, sequence_length, channels, height, width)
		Output size: (batch_size, sequence_length, 6) e.g. torch.Size([16, 20, 6])
		'''

		b,s,c,h,w = hist.size()
		hist = hist.view(b*s,c,h,w)
		state = state.view(b*s,-1)
		h = self.conv(hist).view(-1, self.conv_out)
		h = self.fc(h)
		h = torch.cat((h, state), dim = 1)
		#h = h.permute(0,2,1)
		#h = F.pad(h, (self.width,0))
		#actions = self.t_cnn(h)
		#actions = actions.permute(0,2,1)
		actions = self.fc2(h)
		actions = actions.view(b,s,-1)
		return actions

	def policy(self):
		return Policy(self)
