import keras
from sklearn.metrics import roc_auc_score
import sys
import matplotlib.pyplot as plt
from keras.models import Model
import numpy as np
from keras import backend as K


class FineTuning(keras.callbacks.Callback):
	def __init__(self, startEpoch):
		self.startEpoch = startEpoch

	def on_train_begin(self, logs={}):
		return
	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		if epoch == self.startEpoch:
			self.model.layers[1].trainable = False
			#LR = K.get_value(self.model.optimizer.lr)
			#K.set_value(self.model.optimizer.lr,LR*0.1)
		return

	def on_epoch_end(self, epoch, logs={}):
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return


class DecayLearningRate(keras.callbacks.Callback):
	def __init__(self, startEpoch):
		self.startEpoch = startEpoch

	def on_train_begin(self, logs={}):
		return
	def on_train_end(self, logs={}):
		return

	def on_epoch_begin(self, epoch, logs={}):
		if epoch == self.startEpoch[0] or epoch == self.startEpoch[1]:
			LR = K.get_value(self.model.optimizer.lr)
			K.set_value(self.model.optimizer.lr,LR*0.1)
		return

	def on_epoch_end(self, epoch, logs={}):
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return
