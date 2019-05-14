import numpy as np
import tensorflow as tf

class Reconstruction:

	def __init__(self, image_width=0, image_height=0):
		self.image_width = image_width
		self.image_height = image_height
		# Create the encoder
		encoder = tf.estimator.Estimator(model_fn=self.encoder_model, model_dir="/tmp/encoder_model")
		# Trains the encoder - need to format input
		# encoder.train(input_fn=train_input_fn,steps=1,hooks=[logging_hook])
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

