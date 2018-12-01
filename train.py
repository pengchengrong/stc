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
iterations_per_period = 100
calling_frequency = 10
prev_period_accuracy = 1
total_accuracy = .0
degrading_periods = 0
model_of_last_period = {}
trainning_model = os.path.join(dirname, 'model.training.th')

def auto_early_stopping(iteration, accuracy, model):
	#accuracy is actually validation loss. Smaller means better
	global total_accuracy
	global prev_period_accuracy
	global model_of_last_period
	global degrading_periods
	global trainning_model
	accuracy_plateaued = False
	if iteration !=0 and iteration % iterations_per_period == 0:
		period_accuracy = total_accuracy / (iterations_per_period / calling_frequency) # cause we call this function every calling_frequency iterations
		if period_accuracy > prev_period_accuracy:
			degrading_periods = degrading_periods + 1
			accuracy_plateaued = True
			print('Warning! Validation loss %f'%period_accuracy, ' is higher than all-time low %f'%prev_period_accuracy)
			if degrading_periods > 3:
				print('Recommend early stopping because the performance is worse than all-time low for 4 consective periods.')
				return True, accuracy_plateaued, model_of_last_period
		else:
			degrading_periods = 0
			model_of_last_period = model
			trained_model = {}
			if torch.cuda.is_available():
				cpu_model_dict = {}
				for x in model_of_last_period:
					cpu_model_dict[x] = model_of_last_period[x].cpu()
				trained_model = cpu_model_dict
			else:
				trained_model = model_of_last_period

			torch.save(trained_model, trainning_model)
			print("Validation loss %f"%period_accuracy, " is lower than previously seen low %f"%prev_period_accuracy)
			if prev_period_accuracy < 1 and (prev_period_accuracy - period_accuracy) / prev_period_accuracy < 0.01:
				#accuracy rate increased less than 1 percent during this period
				accuracy_plateaued = True
			prev_period_accuracy = period_accuracy
		total_accuracy = .0		
	total_accuracy = total_accuracy + accuracy
	return False, accuracy_plateaued, None

def train(max_iter, batch_size=1, log_dir=None, aggre = None):
	'''
	This is the main training function, feel free to modify some of the code, but it
	should not be required to complete the assignment.
	'''

	"""
	Load the training data
	"""
	if aggre is not None:
		train_dataloader = load('train', num_workers=0, batch_size=batch_size, subset = aggre)
	else:
		train_dataloader = load('train', num_workers=0, batch_size=batch_size)
	valid_dataloader = load('val', num_workers=0, crop=64, batch_size=batch_size)

	train_dataloader_iterator = cycle(train_dataloader)
	valid_dataloader_iterator = cycle(valid_dataloader)

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
		1., 10., 10., 1., 3., 3.
	])
	# Loss criterion. Your need to replace this with one that considers class imbalance
	sig = nn.Sigmoid()
	loss = nn.BCELoss(weight=gpu(torch.from_numpy(train_class_loss_weights).float()))
	
	for t in range(max_iter):
		batch_obs, batch_states, batch_actions = next(train_dataloader_iterator)
		batch_obs = gpu(batch_obs.float().permute(0,1,4,3,2))
		batch_states = gpu(batch_states.float())
		batch_actions = gpu(batch_actions.float())
		
		model.train()
		
		# zero the gradients (part of pytorch backprop)
		optimizer.zero_grad()
		
		# Compute the model output and loss (view flattens the input)
		model_outputs = model(batch_obs, batch_states)
		predicted_actions = sig(model_outputs)
		#print (predicted_actions[:, 0:10, :] - batch_actions[:, 0:10])

		# Compute the loss
		t_loss_val = loss(predicted_actions, batch_actions)
		
		if t % 50 == 0 and t > 0:
			torch.set_printoptions(precision=2)
			print("train prediction:")
			print (predicted_actions[:, 0:5, :])
			print("train labels:")
			print (batch_actions[:, 0:5])

		# Compute the gradient
		t_loss_val.backward()
	
		# Update the weights
		optimizer.step()

		if t % 10 == 0 and t > 0:	
			batch_obs, batch_states, batch_actions = next(valid_dataloader_iterator)
			batch_obs = gpu(batch_obs.float().permute(0,1,4,2,3))
			batch_states = gpu(batch_states.float())
			batch_actions = gpu(batch_actions.float())
			
			model_outputs = model(batch_obs, batch_states)
			predicted_actions = sig(model_outputs)

			v_loss_val = loss(predicted_actions, batch_actions)

			if t % 50 == 0 and t > 0:
				torch.set_printoptions(precision=2)
				print("val prediction:")
				print (predicted_actions[:, 0:5, :])
				print("val labels:")
				print (batch_actions[:, 0:5])

			print('[%5d]  t_loss = %f   v_loss = %f'%(t, t_loss_val,v_loss_val))
			if log is not None:
				log.add_scalar('train/loss', t_loss_val, t)
				log.add_scalar('val/loss', v_loss_val, t)

			early_stop, accuracy_plateaued, last_good_model = auto_early_stopping(t, v_loss_val, model.state_dict())
			if (early_stop == True):
				model.load_state_dict(last_good_model)
				break
			elif accuracy_plateaued == True and learning_rate > 1e-4:
				learning_rate /= 2.0
				for param_group in optimizer.param_groups:
					param_group['lr'] = learning_rate
				print('learning rate dropped to %f'%learning_rate)
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
