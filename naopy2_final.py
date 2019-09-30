from __future__ import print_function
from keras.models import model_from_json
import sounddevice as sd
from librosa.feature import mfcc as mf
from time import sleep
import os
import numpy as np
from naoqi import ALProxy
import warnings
warnings.filterwarnings("ignore")

ip = "172.16.21.204"
port = 9559

DATA_PATH = "./eng_data/"

def confidence(new_sample, model):

	new_sample = new_sample.reshape(-1)
	# print("Ye: " + str(new_sample.shape))
	sample = array2mfcc(new_sample)
	sample_reshaped = sample.reshape(1,20,11,1)
	u = model.predict(sample_reshaped)
	return u, get_labels()[np.argmax(model.predict(sample_reshaped))]

def get_conf(inp, model):
	labs = get_labels()
	inp = np.array(inp)
	confs, word = confidence(inp, model=model)
	for lab,conf in zip(labs,confs[0]):
	    print(lab , "    " ,conf)
	# print(word)

def get_labels(path=DATA_PATH):
	# labels = os.listdir(path)
	# print(labels)
	labels = ['down', 'go', 'left', 'on', 'right', 'up', 'yes']
	return labels


## Loading classifier

def load_model():
	# load json and create model
	json_file = open('model73.json', 'r')
	model_json = json_file.read()
	json_file.close()
	model = model_from_json(model_json)
	# load weights into new model
	model.load_weights("model73.h5")
	print("Loaded model from disk")
	print("Words to command: ", get_labels())
	return model


## Converting input array to MFCC

def array2mfcc(wave, max_len=11):
	sr = 16000
	wave = wave[::3]
	mfcc = mf(wave, sr=16000)

	# If maximum length exceeds mfcc lengths then pad the remaining ones
	if (max_len > mfcc.shape[1]):
		pad_width = max_len - mfcc.shape[1]
		mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

	# Else cutoff the remaining parts
	else:
		mfcc = mfcc[:, :max_len]
	
	return mfcc


## Record speech

def record():
	duration = 1.5  # seconds
	fs = 16000
	print("Speak for "+ str(duration) + " sec...")
	new_sample = sd.rec(int(duration * fs), samplerate=fs, channels=1)
	sleep(duration+0.3)
	print("Recorded!")
	sd.play(new_sample, fs)
	sleep(duration+0.3)

	return new_sample


## Predict the recorded sample

def predict(model, new_sample):
	# print(new_sample.shape)
	new_sample = new_sample.reshape(-1)
	# print("Ye: " + str(new_sample.shape))
	sample = array2mfcc(new_sample)
	sample_reshaped = sample.reshape(1,20,11,1)
	return get_labels()[np.argmax(model.predict(sample_reshaped))]


## Initalizing the model and instantiating nao procedures

def init():
	model = load_model()

	posture = ALProxy("ALRobotPosture",ip,port)
	navigation = ALProxy("ALNavigation",ip,port)
	tts = ALProxy("ALTextToSpeech",ip,port)
	motion = ALProxy("ALMotion",ip,port)
	battery = ALProxy("ALBattery",ip,port)
	led = ALProxy("ALLeds",ip,port)

	nao_com = [posture, navigation, tts, motion, battery, led]

	return model, nao_com


## Call to nao robot

def nao_call(word, nao_com):
	posture, navigation, tts, motion, battery, led = nao_com

	try:
		heard = "I heard: "+ word
		tts.say(heard)
		if(word == 'down'):
			posture.goToPosture("SitRelax",1.0)
		if(word == 'up'):
			posture.goToPosture("StandInit",1.0)
		if(word == 'left'):
			motion.moveTo(0.0,0.0,1.57)
		if(word == 'right'):
			motion.moveTo(0.0,0.0,-1.57)
		if(word == 'yes'):
			tts.say("I am happy")
		if(word == 'no'):
			tts.say("I am sad")
		if(word == 'go'):

			tts.say("NAO is in motion")
			# navigation.navigateTo(0.5,0.5)
			motion.moveTo(0.2,0,0)
		if(word == 'stop'):
			motion.stopMove()
		if(word == 'on'):
			x = battery.getBatteryCharge()
			x = str(x)
			x = "The battery is "+ str(x) + "%"
			tts.say(x)
		if(word == 'off'):
			# tts.say("No no no yein yein yein!")
			tts.say("No no no!")
			led.rasta(3.0)
		if(word == '_background_noise_'):
			pass
	except Exception as e:
		print(e)


model, nao_com = init()
# model = init()

while(True):

	print()
	print("Press Enter to continue...")
	raw_input()

	# Emergency stop

	# inp = input()
	# if(inp == "xxx"):
	# 	motion.stopMove()

	sample = record()
	predicted = predict(model, sample)
	print("Predicted value is: " + predicted)
	print()
	print(get_conf(sample, model))
	print()

	nao_call(predicted, nao_com)

	# print("YEYEYEYEYEYE")
	# print("Fuck you. Be attentive!!!!!! ")

	print()
	












