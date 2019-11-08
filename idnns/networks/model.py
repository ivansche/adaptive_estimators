import functools
import tensorflow as tf
import numpy as np
from idnns.networks.utils import _convert_string_dtype
from idnns.networks.models import multi_layer_perceptron
from idnns.networks.models import deepnn
from idnns.networks.ops import *


def lazy_property(function):
	attribute = '_cache_' + function.__name__

	@property
	@functools.wraps(function)
	def decorator(self):
		# print hasattr(self, attribute)
		if not hasattr(self, attribute):
			setattr(self, attribute, function(self))
		return getattr(self, attribute)

	return decorator

def prelu(_x, name=''):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

def srelu(_x, name=''):
  _alpha = tf.get_variable('alpha', _x.get_shape()[-1],
						   initializer=tf.constant_initializer(5.0),
						   dtype=tf.float32)
  _beta = tf.get_variable('beta', _x.get_shape()[-1],
						   initializer=tf.constant_initializer(1.0),
						   dtype=tf.float32)
  print(_alpha)
  first = tf.maximum(0.0, _x)
  first = tf.minimum(_alpha, first)
  second = _beta * (tf.maximum(_alpha, _x) - _alpha)
  return  first + second

def swish(_x, name=''):
    return _x*tf.nn.sigmoid(_x)

def myabs(_x,name=''):
    return tf.abs(_x)

class Model:
	"""A class that represent model of network"""

	def __init__(self, input_size, layerSize, num_of_classes, learning_rate_local=0.001, save_file='',
	             activation_function=0,batch_norm=False, cov_net=False, l2=False, penalty=0.01):
		self.covnet = cov_net
		self.l2reg=l2
		self.batch_norm = batch_norm
		self.input_size = input_size
		self.layerSize = layerSize
		self.all_layer_sizes = np.copy(layerSize)
		self.all_layer_sizes = np.insert(self.all_layer_sizes, 0, input_size)
		self.num_of_classes = num_of_classes
		self._num_of_layers = len(layerSize) + 1
		self.learning_rate_local = learning_rate_local
		self._save_file = save_file
		self.hidden = None
		self.savers = []
		self.he=1
		self.l2_penalty=penalty
		self.af=activation_function
		if activation_function == 1:
			self.activation_function = tf.nn.relu
			self.he=2.0
		elif activation_function == 2:
			self.activation_function = tf.nn.softplus
			self.he=2.0
		elif activation_function == 3:
			self.activation_function = tf.nn.elu
			self.he=2.0
		elif activation_function == 4:
			self.activation_function = prelu
			self.he=2.0
		elif activation_function == 5:
			self.activation_function = swish
			self.he=2.0
		elif activation_function == 6:
			self.activation_function = myabs
		elif activation_function == 10:
			self.he=2.0
			self.activation_function = srelu
		else:
			self.activation_function = tf.nn.tanh
		self.prediction
		self.optimize
		self.accuracy

	def initilizae_layer(self, name_scope, row_size, col_size, activation_function, last_hidden):
		# Bulid layer of the network with weights and biases
		weights = get_scope_variable(name_scope=name_scope, var="weights",
		                             shape=[row_size, col_size],
		                             initializer=tf.truncated_normal_initializer(mean=0.0, stddev=1.0*self.he / np.sqrt(
			                             float(row_size))))
		biases = get_scope_variable(name_scope=name_scope, var='biases', shape=[col_size],
		                            initializer=tf.constant_initializer(0.0))

		self.weights_all.append(weights)
		self.biases_all.append(biases)
		variable_summaries(weights)
		variable_summaries(biases)
		with tf.variable_scope(name_scope) as scope:
			if self.batch_norm and (col_size==self.layerSize[-1]):
        			input = tf.matmul(last_hidden, weights) + biases
        			if activation_function==None:
        				prebn=input
        			else:
        				prebn=activation_function(input, name='output')
        			output=tf.layers.batch_normalization(prebn,momentum=0.99, training=self.phase)

			else:
        			input = tf.matmul(last_hidden, weights) + biases
        			if activation_function == None:
        				output = input
        			else:
        				output = activation_function(input, name='output')
        				#if self.af==2:
        				#output=output1-np.log(2)
		self.inputs.append(input)
		self.hidden.append(output)
		return output

		'''

        input = tf.matmul(last_hidden, weights) + biases
        			bn = tf.layers.batch_normalization(input,momentum=0.99, training=self.phase)
        			if activation_function == None:
        				output = tf.identity(bn, name='output')
        			else:
        				output = activation_function(bn, name='output')

        			input = tf.matmul(last_hidden, weights) + biases
        			bn = tf.layers.batch_normalization(input,momentum=0.99, training=self.phase)
        			if activation_function == None:
        				output = tf.identity(bn, name='output')
        			else:
        				output = activation_function(bn, name='output')

                        input = tf.matmul(last_hidden, weights) + biases
        			if activation_function == None:
        				prebn = input
        			else:
        				prebn = activation_function(input, name='output')
        			output = tf.layers.batch_normalization(prebn, momentum=0.98, training=self.phase)
		'''
	@property
	def num_of_layers(self):
		return self._num_of_layers

	@property
	def hidden_layers(self):
		"""The hidden layers of the netowrk"""
		if self.hidden is None:
			self.hidden, self.inputs, self.weights_all, self.biases_all = [], [], [], []
			last_hidden = self.x
			if self.covnet == 1:
				y_conv, self._drouput, self.hidden, self.inputs = deepnn(self.x)
			elif self.covnet == 2:
				y_c, self.hidden, self.inputs = multi_layer_perceptron(self.x, self.input_size, self.num_of_classes,
				                                                       self.layerSize[0], self.layerSize[1])
			else:

				self._drouput = 'dr'
				# self.hidden.append(self.x)
				for i in range(1, len(self.all_layer_sizes)):
					name_scope = 'hidden' + str(i - 1)
					row_size, col_size = self.all_layer_sizes[i - 1], self.all_layer_sizes[i]
					activation_function = self.activation_function
					last_hidden = self.initilizae_layer(name_scope, row_size, col_size, activation_function,
					                                    last_hidden)
				name_scope = 'final_layer'
				row_size, col_size = self.layerSize[-1], self.num_of_classes
				activation_function = None
				last_hidden = self.initilizae_layer(name_scope, row_size, col_size, activation_function, last_hidden)
		return self.hidden

	@lazy_property
	def prediction(self):
		logits = self.hidden_layers[-1]
		return logits

	@lazy_property
	def drouput(self):
		return self._drouput

	@property
	def optimize(self):
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		print(tf.all_variables())
		with tf.control_dependencies(update_ops):
				optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_local).minimize(self.cross_entropy)
				return optimizer

	@lazy_property
	def x(self):
		return tf.placeholder(tf.float32, shape=[None, self.input_size], name='x')

	@lazy_property
	def labels(self):
		return tf.placeholder(tf.float32, shape=[None, self.num_of_classes], name='y_true')

	@lazy_property
	def phase(self):
		return tf.placeholder(tf.bool, name='phase')

	@lazy_property
	def accuracy(self):
		correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		tf.summary.scalar('accuracy', accuracy)
		return accuracy

	@lazy_property
	def cross_entropy(self):
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.labels,logits=self.prediction)
		if self.l2reg:
			l2loss=self.l2_penalty*sum([tf.nn.l2_loss(self.weights_all[i]) for i in range(len(self.weights_all)-1)])
			loss=l2loss+cross_entropy
			tf.summary.scalar('cross_entropy', loss)
			return loss
		else:
			tf.summary.scalar('cross_entropy', cross_entropy)
			return cross_entropy

	@property
	def save_file(self):
		return self._save_file
	'''
	def inference(self, data):
		"""Return the predication of the network with the given data"""
		with tf.Session() as sess:
			self.saver.restore(sess, './' + self.save_file)
			feed_dict = {self.x: data}
			pred = sess.run(self.prediction, feed_dict=feed_dict)
		return pred

	def inference_default(self, data):
		session = tf.get_default_session()
		feed_dict = {self.x: data}
		pred = session.run(self.prediction, feed_dict=feed_dict)
		return pred

	def get_layer_with_inference(self, data, layer_index, epoch_index):
		"""Return the layer activation's values with the results of the network"""
		with tf.Session() as sess:
			self.savers[epoch_index].restore(sess, './' + self.save_file + str(epoch_index))
			feed_dict = {self.hidden_layers[layer_index]: data[:, 0:self.hidden_layers[layer_index]._shape[1]]}
			pred, layer_values = sess.run([self.prediction, self.hidden_layers[layer_index]], feed_dict=feed_dict)
		return pred, layer_values

	def calc_layer_values(self, X, layer_index):
		"""Return the layer's values"""
		with tf.Session() as sess:
			self.savers[-1].restore(sess, './' + self.save_file)
			feed_dict = {self.x: X}
			layer_values = sess.run(self.hidden_layers[layer_index], feed_dict=feed_dict)
		return layer_values

	def update_weights_and_calc_values_temp(self, d_w_i_j, layer_to_perturbe, i, j, X):
		"""Update the weights of the given layer cacl the output and return it to the original values"""
		if layer_to_perturbe + 1 >= len(self.hidden_layers):
			scope_name = 'softmax_linear'
		else:
			scope_name = "hidden" + str(layer_to_perturbe)
		weights = get_scope_variable(scope_name, "weights", shape=None, initializer=None)
		session = tf.get_default_session()
		weights_values = weights.eval(session=session)
		weights_values_pert = weights_values
		weights_values_pert[i, j] += d_w_i_j
		set_value(weights, weights_values_pert)
		feed_dict = {self.x: X}
		layer_values = session.run(self.hidden_layers[layer_to_perturbe], feed_dict=feed_dict)
		set_value(weights, weights_values)
		return layer_values

	def update_weights(self, d_w0, layer_to_perturbe):
		"""Update the weights' values of the given layer"""
		weights = get_scope_variable("hidden" + str(layer_to_perturbe), "weights", shape=None, initializer=None)
		session = tf.get_default_session()
		weights_values = weights.eval(session=session)
		set_value(weights, weights_values + d_w0)

	def get_wights_size(self, layer_to_perturbe):
		"""Return the size of the given layer"""
		weights = get_scope_variable("hidden" + str(layer_to_perturbe), "weights", shape=None, initializer=None)
		return weights._initial_value.shape[1].value, weights._initial_value.shape[0].value

	def get_layer_input(self, layer_to_perturbe, X):
		"""Return the input of the given layer for the given data"""
		session = tf.get_default_session()
		inputs = self.inputs[layer_to_perturbe]
		feed_dict = {self.x: X}
		layer_values = session.run(inputs, feed_dict=feed_dict)
		return layer_values
	'''
