import numpy as np
import tensorflow as tf
import sys
import os
import cv2

class Reconstruction:

	def __init__(self, image_width=0, image_height=0, train_directory=''):
		self.image_width = image_width
		self.image_height = image_height
		train_input = self.load_images(train_directory)
		# Create the encoder
		encoder = tf.estimator.Estimator(model_fn=self.encoder_model, model_dir="/tmp/encoder_model")
		# Trains the encoder - need to format input
		# encoder.train(input_fn=train_input_fn,steps=1,hooks=[logging_hook])
		print("Initailiazed class object")
		pass

	def load_images(self, directory):
		'''
			Usage: Loads images for training. Currently takes a subset of the images of size 3000 for testing the network.
				The directory needs to be in the same directory as main.py
			Parameters:
				- directory: name of the directory with the images for training
			Returns:
				- images: an array of numpy arrays where each numpy array is the read image
		'''
		curr_path = os.getcwd()
		if not os.path.exists(directory):
			print("Directory [" + directory + "] does not exist!")
			sys.exit(1)

		images = []
		image_path = curr_path + '/' + directory
		currently_loaded = 0
		limit = False
		for r, d, f in os.walk(image_path):
			for file in f:
				if '.png' in file:
					img = cv2.imread(os.path.join(r,file))
					images.append(img)
					currently_loaded += 1
					if (currently_loaded >= 3000):
						limit = True
						break
			if (limit):
				break
		return images

	def encoder_model(self, features, labels, mode):
		'''
			Usage: Model function defined for the 2D convolutional encoder of the network that is passed to
				a tensorflow estimator class
			Parameters:
				- features: feature data to be used to train
				- labels: labels of the feature data
				- mode: enum from tf.estimator.ModeKeys for estimating loss and optimizing
		'''

		# may need to do some fine-tuning of a lot of these parameters
		num_filters = 32

		# Input Layer
		input_layer = tf.reshape(features["x"], [-1, self.image_width, self.image_height, 3])

		# Convolutional Layer #1
		conv1 = tf.layers.conv2d(
			inputs = input_layer,
			filters = num_filters,
			kernel_size = [7,7],
			padding = "same",
			activation = tf.nn.relu
		)

		# Pooling Layer #1
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

		# Convolutional Layer #2
		conv2 = tf.layers.conv2d(
			inputs = pool1,
			filters = num_filters,
			kernel_size = [3,3],
			padding = "same",
			activation = tf.nn.relu
		)
		# Pooling Layer #2
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

		conv3 = tf.layers.conv2d(
			inputs = pool2,
			filters = num_filters,
			kernel_size = [3,3],
			padding = "same",
			activation = tf.nn.relu
		)
		pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)

		conv4 = tf.layers.conv2d(
			inputs = pool3,
			filters = num_filters,
			kernel_size = [3,3],
			padding = "same",
			activation = tf.nn.relu
		)
		pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=2)

		conv5 = tf.layers.conv2d(
			inputs = pool4,
			filters = num_filters,
			kernel_size = [3,3],
			padding = "same",
			activation = tf.nn.relu
		)
		pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2,2], strides=2)

		conv6 = tf.layers.conv2d(
			inputs = pool5,
			filters = num_filters,
			kernel_size = [3,3],
			padding = "same",
			activation = tf.nn.relu
		)
		pool6 = tf.layers.max_pooling2d(inputs=conv6, pool_size=[2,2], strides=2)

		# Fully Connected Layer
		dim = 32
		flatten = tf.reshape(pool6, [-1, dim * dim * dim])
		fc_layer = tf.layers.dense(inputs=flatten, units=256, activation=tf.nn.relu)
		dropout = tf.layers.dropout(
			inputs = fc_layer, 
			rate = 0.4, 
			training = mode == tf.estimator.ModeKeys.TRAIN
		)

