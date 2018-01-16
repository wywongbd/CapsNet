import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt


class CapsuleNet:
	def __init__(self, session, optimizer=tf.train.AdamOptimizer):
		# tensorflow session
		self.sess = session

		# MNIST training data, expected shape is [batch_size, 28, 28, 1]
		self.imgs = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="imgs")

		# MNIST labels
		self.labels = tf.placeholder(shape=[None], dtype=tf.int64, name="labels")
		self.labels_one_hot = tf.one_hot(self.labels, depth=10, name="labels_one_hot")

		# batch size
		self.batch_size = (self.imgs).get_shape()[0]

		# Chosen optimzer for training
		self.optimizer = tf.train.AdamOptimizer()

		# Keep track of all layers in a python dictionary
		self.layers = {}

		# Keep track of manually initialized weights
		self.weights = {}

		# Construct tensorflow computation graph
		self.build_model()

		# Prepare for training procedure
		self.init_optimizer()

	def build_model(self):
		# initialize local dictionary 
		layers = {}
		weights = {}
		self.batch_size = tf.shape(self.imgs)[0]

		# First two layers are convolutional layers
		# Output shape:
		# conv1 -> [batch_size, (28-9+0)/1 + 1, (28-9+0)/1 + 1, 256] = [batch_size, 20, 20, 256]
		# conv2 -> [batch_size, (20-9+0)/2 + 1, (20-9+0)/2 + 1, 256] = [batch_size, 6, 6, 256] 		
		layers['conv1'] = tf.layers.conv2d(self.imgs, name="conv1", filters=256, kernel_size=9, strides=(1,1), padding="valid", activation=tf.nn.relu)
		layers['conv2'] = tf.layers.conv2d(layers['conv1'], name="conv2", filters=256, kernel_size=9, strides=(2,2), padding="valid", activation=tf.nn.relu)

		# first capsule layer is fully connected to second layer, each capsule outputs a vector of length 8, so we can flatten the output of conv2
		layers['conv2_flattened'] = tf.reshape(layers['conv2'], [-1, 1152, 8], name="caps1_raw")

		# apply the squash function to ensure that the size of the vectors are between 0 and 1
		layers['caps1'] = self.squash(layers['conv2_flattened']) 

		# initialize weights for 'predicting' outputs of second capsule layer 
		W_init = tf.truncated_normal(shape=[1, 1152, 10, 16, 8], stddev=0.01, dtype=tf.float32, name="W_init")
		weights['W'] = tf.Variable(W_init, name="W")
		weights['W_tiled'] = tf.tile(weights['W'], [self.batch_size, 1, 1, 1, 1], name="W_tiled")

		# expand and tile 
		layers['caps1_expanded'] = tf.expand_dims(tf.expand_dims(layers['caps1'], -1), 2, name="caps1_expanded")
		layers['caps1_expanded_tiled'] = tf.tile(layers['caps1_expanded'], [1, 1, 10, 1, 1], name="caps1_expanded_tiled")

		# multiply
		layers['caps2_predicted'] = tf.matmul(weights['W_tiled'], layers['caps1_expanded_tiled'], name="caps2_predicted")
		
		# Routing by agreement
		iteration = tf.constant(0)
		dummy_init = tf.truncated_normal(shape=[tf.shape(self.imgs)[0], 1, 10, 16, 1], stddev=0.01, dtype=tf.float32, name='dummy_init')
		raw_weights = tf.zeros([self.batch_size, 1152, 10, 1, 1], dtype=np.float32, name="raw_weights")
		raw_weights, caps2_predicted, layers['caps2_output'], iteration = tf.while_loop(self.routing_condition, 
															self.routing_loop_body, 
															[raw_weights, layers['caps2_predicted'], dummy_init, iteration])

		# Compute class scores
		layers['y_prob'] = self.safe_norm(s=layers['caps2_output'], axis=-2)
		layers['y_pred'] = tf.squeeze(tf.argmax(layers['y_prob'], axis=2), axis=[1,2], name='y_pred')
		
		layers['caps2_output_norm'] = self.safe_norm(layers['caps2_output'], axis=-2, keep_dims=True, name="caps2_output_norm")

		# Calculate marginal loss
		present_error_raw = tf.square(tf.maximum(0., 0.9 - layers['caps2_output_norm']), name="present_error_raw")
		present_error = tf.reshape(present_error_raw, shape=(-1, 10), name="present_error")
		absent_error_raw = tf.square(tf.maximum(0., layers['caps2_output_norm'] - 0.1), name="absent_error_raw")
		absent_error = tf.reshape(absent_error_raw, shape=(-1, 10), name="absent_error")

		layers['L'] = tf.add(self.labels_one_hot * present_error, 0.5 * (1.0 - self.labels_one_hot) * absent_error, name="L")
		self.margin_loss = tf.reduce_mean(tf.reduce_sum(layers['L'], axis=1), name="margin_loss")

		# According to the paper, during the training procedure, only the output vector of the capsule that corresponds to the target digit is sent to decoder
		# We need to 'mask' other values
		self.mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")
		reconstruction_targets = tf.cond(self.mask_with_labels, lambda: self.labels, lambda: layers['y_pred'], name="reconstruction_targets")
		reconstruction_mask = tf.one_hot(reconstruction_targets, depth=10, name="reconstruction_mask")
		reconstruction_mask_reshaped = tf.reshape(reconstruction_mask, [-1, 1, 10, 1, 1], name="reconstruction_mask_reshaped")
		caps2_output_masked = tf.multiply(layers['caps2_output'], reconstruction_mask_reshaped, name="caps2_output_masked")
				
		layers['decoder_input'] = tf.reshape(caps2_output_masked, [-1, 160], name="decoder_input")

		# Two fully connected hidden layers with 512 and 1024 neurons
		layers['hidden1'] = tf.layers.dense(layers['decoder_input'], 512, activation=tf.nn.relu, name="hidden1")
		layers['hidden2'] = tf.layers.dense(layers['hidden1'], 1024, activation=tf.nn.relu, name="hidden2")
		layers['decoder_output'] = tf.layers.dense(layers['hidden2'], 28*28, activation=tf.nn.sigmoid, name="decoder_output")

		# Compute reconstruction loss
		imgs_flat = tf.reshape(self.imgs, [-1, 28*28], name="imgs_flat")
		squared_difference = tf.square(imgs_flat - layers['decoder_output'], name="squared_difference")
		self.reconstruction_loss = tf.reduce_sum(squared_difference, name="reconstruction_loss")

		self.loss = tf.add(self.margin_loss, 0.005 * self.reconstruction_loss, name="loss")

		self.layers = layers
		self.weights = weights

	def init_optimizer(self):
		correct = tf.equal(self.labels, self.layers['y_pred'], name="correct")
		self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

		self.training_op = (self.optimizer).minimize(self.loss, name="training_op")

	def routing_condition(self, raw_weights, caps2_predicted, caps2_output, counter):
		# routing algorithm is set to run 3 iterations
		return tf.less(counter, 3)

	def routing_loop_body(self, raw_weights, caps2_predicted, caps2_output, counter):
		# loop body of routing by agreement algorithm
		routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")
		weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
		weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum")

		caps2_output = self.squash(weighted_sum, axis=-2)
		
		return raw_weights, caps2_predicted, caps2_output, tf.add(counter, 1)

	def squash(self, s, axis=-1, eps=1e-7):
		squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)

		# epsilon to prevent division by zero
		safe_norm = tf.sqrt(squared_norm + eps)
		squash_factor = squared_norm / (1. + squared_norm)
		unit_vector = s / safe_norm

		return squash_factor * unit_vector

	def safe_norm(self, s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
		squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)

		# epsilon to prevent division by zero in later steps
		return tf.sqrt(squared_norm + epsilon)