import numpy as np
import tensorflow as tf
import h5py

class Reconstruction:

	def __init__(self, image_width=0, image_height=0):
		self.image_width = image_width
		self.image_height = image_height
		print("Initailiazed class object")
		pass

	def encoder_model(self, features, labels, mode):
		'''
			Usage: Model function defined for the 2D convolutional encoder of the network that is passed to
				a tensorflow estimator class
			Parameters:
				- features: feature data to be used to train
				- labels: labels of the feature data
				- mode: enum from tf.estimator.ModeKeys for estimating loss and optimizing
		'''

		# Input Layer
		input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
		pass
