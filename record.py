import os, uuid
import json
import base64
import datetime
import _thread
import torch
import glob
import argparse
import train
from torch import nn, optim
from time import gmtime, strftime
from pykart import Kart
from time import sleep, time
from random import random
from pynput import keyboard
from models import *
from pylab import *

# Make sure we can record
try:
	os.mkdir('train')
except:
	pass


folder_name = None
terminated = False
restart = False
recording_index = -1
recorded_files = 0
state = {'smooth_speed': 0.0, 'energy': 0.0, 'speed': 0.0, 'wrongway': 0.0, 'position_along_track': 0.0, 'angle': 0.0, 'finish_time': 0.0, 'position_in_race': 4.0, 'timestamp': 0.0, 'distance_to_center': 0.0}
dirname = os.path.dirname(os.path.abspath(__file__))
model = Model()
if model is None:
	print( "Failed to load model. Did you train one?" )
	exit(1)
trainning_model = os.path.join(dirname, 'model.th')
if os.path.exists(trainning_model):
	model.load_state_dict(torch.load(trainning_model))
	print("Pre-trained model loaded.")
model.eval()
print ("Num of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))

def on_press(key):
	global fire, nitro, brake, fwd, right, left
	global terminated, restart
	try:
		if key.char == 'q':
			terminated = True
			print("Terminated")
		if key.char == 'r':
			restart = True
		elif key == keyboard.Key.n:
			nitro = 1
	except AttributeError:
		if key == keyboard.Key.left:
			left = 1
		elif key == keyboard.Key.right:
			right = 1
		elif key == keyboard.Key.up:
			fwd = 1
		elif key == keyboard.Key.down:
			brake = 1
		elif key == keyboard.Key.space:
			fire = 1
			
	return True

def on_release(key):
	global fire, nitro, brake, fwd, right, left

	try:
		if key == keyboard.Key.n:
			nitro = 0
	except AttributeError:
		if key == keyboard.Key.left:
			left = 0
		elif key == keyboard.Key.right:
			right = 0
		elif key == keyboard.Key.up:
			fwd = 0
		elif key == keyboard.Key.down:
			brake = 0
		elif key == keyboard.Key.space:
			fire = 0
	return True

def get_action(logits):
	probs = 1. / (1. + np.exp(-logits.detach().numpy()))
	bits = np.array([np.random.uniform()<=p for p in probs]).astype(int)
	return int(np.dot(bits, [32, 16, 8, 4, 2, 1]))

def extract_state(state):
	s = torch.zeros(8)
	s[0] = state['smooth_speed']
	s[1] = state['energy']
	s[2] = state['speed']
	s[3] = state['wrongway']
	s[4] = state['position_along_track']
	s[5] = state['angle']
	s[6] = state['distance_to_center']
	s[7] = state['position_in_race']
	return s

def savedata(data, states, label):
	if data.size()[0] < 40:
		return

	global recording_index,recorded_files
	if recording_index < 0:
		try:
			LatestFile = max(glob.iglob('train/*_img.npy'), key=os.path.getctime)
			recording_index = int(LatestFile[6:10])
		except Exception as e:
			recording_index = -1
		
	
	recording_index += 1
	recorded_files += 1
	fname = format(recording_index , '04d')	

	np.save(os.path.join('train/', fname + "_img.npy"), data.numpy())
	np.save(os.path.join('train/', fname + "_state.npy"), states.numpy())
	np.save(os.path.join('train/', fname + "_action.npy"), label.numpy())


K = Kart("lighthouse", 300, 200)

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--da', type=bool, default=False)
data_aggregation = parser.parse_args().da

while not terminated:
	print ("Game started.")
	K.restart()
	K.waitRunning()

	stuck_frames = 0
	fire, nitro, brake, fwd, right, left  = 0, 0, 0, 0, 0, 0
	prediction = 8 # go forward before the first predicton comes out
	data = None
	label = None
	states = None
	restart = False

	# Collect events until released
	with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
		#listener.join()
		im, lbl = None, None
		policy = model.policy()
		prev_state = None
		prev_obs = None
		prev_img = None
		human_control = False
		finish_time = 0

		while not (terminated == True or restart == True or finish_time > 0):
			action = 128 * fire + 16 * nitro + 8 * brake + 4 * fwd + 2 * right + 1 * left
			if action == 0 or restart:
				#Using model to play
				if human_control == True:
					# The human player just released control
					human_control = False
					savedata(data, states,  label)
					data = None
					label = None
					states = None

				feedback = K.step(prediction)
			else:
				#Human control
				if human_control == False:
					human_control = True
				
				# auto steering
				'''
				sane_distance = 3.
				prev_angle = state["angle"]
				if prev_state["distance_to_center"] < -sane_distance and prev_angle < 0 and (right == 0 or left == 1):
					action = 32 * fire + 16 * nitro + 8 * brake + 4 * fwd + 2 * 1 + 1 * 0
					print("Auto steering right")
				elif prev_state["distance_to_center"] > sane_distance and prev_angle > 0 and (right == 1 or left == 0):
					action = 32 * fire + 16 * nitro + 8 * brake + 4 * fwd + 2 * 0 + 1 * 1
					print("Auto steering left")
				'''

				
				if prev_img is not None: #skip first frame because we havn't made any observation yet.
					d = prev_img
					s = extract_state(prev_state)
					l = torch.tensor([fire, nitro, brake, fwd, right, left])
					#add one dimension for batch
					d = d.view(1, d.size()[0], d.size()[1], d.size()[2])
					s = s.view(1, s.size()[0])
					l = l.view(1, l.size()[0])

					if data  is None:
						data = d
						states = s
						label = l
					else:
						data = torch.cat((data, d), dim = 0)
						states = torch.cat((states, s), dim = 0)
						label = torch.cat((label, l), dim = 0)

				feedback = K.step(action)

			if feedback is not None and feedback[1] is not None:
				state = feedback[0]
				obs = feedback[1]
				finish_time = state["finish_time"]
				
				img = torch.as_tensor(obs).float()
				#img = img.view(1, 1, img.size()[0], img.size()[1], img.size()[2])
				logits = policy(img.permute(2,1,0), extract_state(state))
				prediction = get_action(logits)

				prev_state = state
				prev_obs = obs
				prev_img = img

				'''
				ion()
				if obs is not None:
					if im is None:
						im = imshow(obs)
					else:
						im.set_data(obs)
				draw()
				'''
				
			pause(0.1)

		if human_control == True:
			# The human player just finished a race
			human_control = False
			savedata(data, states,  label)
			data = None
			label = None
			states = None

		if data_aggregation == True:
			#Train with newly collected data
			aggre = []
			for i in range (recorded_files):
				aggre.append(format(recording_index - i, '04d')+'_img.npy')
			train(1, subset = aggre)

torch.save(model.state_dict(), os.path.join(dirname, 'model.th')) 
quit()
	