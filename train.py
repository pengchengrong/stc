import argparse, pickle
import numpy as np
import os
from itertools import cycle

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from utils import *
from models import *

dirname = os.path.dirname(os.path.abspath(__file__))
trainning_model = os.path.join(dirname, 'model.training.th')

def cycle(seq):
	while True:
		for elem in seq:
			yield elem


def gpu(obj):
	if torch.cuda.is_available():
		return obj.cuda()
	else:
		return obj
'''
def hack_loss(states, labels):
	total_loss = 0

	for i in range(states.size()[0]):
		for j in range(states.size()[1]):
			if states[i][j][6] < -0.5:
				if labels[i][j][4] == 0:
					total_loss += 1
				if labels[i][j][5] == 1:
					total_loss += 1
			elif states[i][j][6] > 0.5:
				if labels[i][j][4] == 1:
					total_loss += 1
				if labels[i][j][5] == 0:
					total_loss += 1

	return torch.tensor(total_loss / states.numel())
'''

def train(max_iter, batch_size=10, log_dir=None, aggre = None):
	'''
	This is the main training function, feel free to modify some of the code, but it
	should not be required to complete the assignment.
	'''

	"""
	Load the training data
	"""
	if aggre is not None:
		train_dataloader = load('train', num_workers=0, crop=24, batch_size=batch_size, subset = aggre)
	else:
		train_dataloader = load('train', num_workers=0, crop=24, batch_size=batch_size)
	#valid_dataloader = load('val', num_workers=0, crop=40, batch_size=batch_size)

	train_dataloader_iterator = cycle(train_dataloader)
	#valid_dataloader_iterator = cycle(valid_dataloader)

	model = gpu(Model())
	print (model)
	if torch.cuda.is_available:
		print ("Working on GPU.")

	print ("Num of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))

	learning_rate = 1e-3
	if os.path.exists(trainning_model):
		model.load_state_dict(torch.load(trainning_model), False)
		learning_rate = 1e-4
		print('Resume trainning from last point')
	else:
		print('Trainning from scratch')

	log = None
	if log_dir is not None:
		from .utils import SummaryWriter
		log = SummaryWriter(log_dir)
	
	# If your model does not train well, you may swap out the optimizer or change the lr
	
	optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=1e-4)
	
	train_class_loss_weights = np.array([
		1., 15., 15., 1., 8., 8.
	])
	# Loss criterion. Your need to replace this with one that considers class imbalance
	weighted_loss = nn.BCEWithLogitsLoss(pos_weight=gpu(torch.from_numpy(train_class_loss_weights).float()))
	loss = nn.BCEWithLogitsLoss()
	
	for t in range(max_iter):
		batch_obs, batch_states, batch_actions = next(train_dataloader_iterator)
		batch_obs = gpu(batch_obs.float().permute(0,1,4,3,2))
		#batch_states = gpu(batch_states.float().permute(0,2,1))
		batch_actions = gpu(batch_actions)
		
		model.train()
		
		# zero the gradients (part of pytorch backprop)
		optimizer.zero_grad()
		
		# Compute the model output and loss (view flattens the input)
		model_outputs = model(batch_obs, batch_states)

		#hloss = hack_loss(batch_states, batch_actions)

		# Compute the loss
		t_loss_val = loss(model_outputs, batch_actions.float())# + hloss
		
		# Compute the gradient
		t_loss_val.backward()
	
		# Update the weights
		optimizer.step()

	
		if t % 10 == 0 and t > 0:	
			print('[%5d]  t_loss = %f'%(t, t_loss_val))
			if log is not None:
				log.add_scalar('train/loss', t_loss_val, t)
				#log.add_scalar('val/loss', v_loss_val, t)
			torch.save(model.state_dict(), trainning_model) 

	torch.save(model.state_dict(), os.path.join(dirname, 'model.th')) # Do NOT modify this line

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--max_iter', type=int, default=3000)
	parser.add_argument('-l', '--log_dir')
	args = parser.parse_args()

	print ('[I] Start training')
	train(args.max_iter, log_dir=args.log_dir)
	print ('[I] Training finished')
