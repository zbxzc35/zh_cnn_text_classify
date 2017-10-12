import tensorflow as tf
import numpy as np


class TextCNN(object):
	'''
    A CNN for text classification
    Uses and embedding layer, followed by a convolutional, max-pooling and softmax layer.
    '''
	def __init__(
			self, sequence_length, num_classes,
			embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

		# Placeholders for input, output, dropout
		self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name = "input_x")
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name = "input_y")
		self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")

		# Keeping track of l2 regularization loss (optional)
		l2_loss = tf.constant(0.0)

		# Embedding layer
		# self.embedded_chars = [None(batch_size), sequence_size, embedding_size]
		# self.embedded_chars = [None(batch_size), sequence_size, embedding_size, 1(num_channels)]
		self.embedded_chars = self.input_x
		self.embedded_chars_expended = tf.expand_dims(self.embedded_chars, -1)

		# Create a convolution + maxpool layer for each filter size
		pooled_outputs = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				# Convolution layer
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
				conv = tf.nn.conv2d(
					self.embedded_chars_expended,
					W,
					strides=[1,1,1,1],
					padding="VALID",
					name="conv")
				# Apply nonlinearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name = "relu")
				# Maxpooling over the outputs
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1,1,1,1],
					padding="VALID",
					name="pool")
				pooled_outputs.append(pooled)

		# Combine all the pooled features
		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(pooled_outputs, 3)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		# Add dropout
		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

		# Final (unnomalized) scores and predictions
		with tf.name_scope("output"):
			W1 = tf.get_variable(
				"W1",
				shape = [num_filters_total, 2],
				initializer = tf.contrib.layers.xavier_initializer())
			b1 = tf.Variable(tf.constant(0.1, shape=[2], name = "b1"))
			l2_loss += tf.nn.l2_loss(W1)
			l2_loss += tf.nn.l2_loss(b1)
			pos_neg_var = tf.nn.xw_plus_b(self.h_drop, W1, b1, name="pos_neg")
			pos_neg = tf.nn.softmax(pos_neg_var)

		with tf.name_scope("neutral"):
			multi_pos_neg = tf.multiply(pos_neg[:,:1],pos_neg[:,1:])
			self.W2 = tf.Variable(tf.constant(4., shape=[1], name="W2"))
			self.b2 = tf.Variable(tf.constant(0.1, shape=[1], name="b2"))
			neutral = tf.add(tf.multiply(multi_pos_neg, self.W2), self.b2, name="pos_neg")
			self.scores = tf.concat([pos_neg[:,:1],neutral, pos_neg[:,1:]],1, name="scores")

			# W2 = tf.get_variable(
			# 	"W2",
			# 	shape = [3, num_classes],
			# 	initializer = tf.contrib.layers.xavier_initializer())
			# b2 = tf.Variable(tf.constant(0.1, shape=[num_classes], name = "b2"))
			# self.scores = tf.nn.xw_plus_b(self.pos_neg_neutral, W2, b2, name = "scores")
			self.predictions = tf.argmax(self.scores, 1, name = "predictions")
			self.training_scores = tf.concat([pos_neg[:, :1], tf.zeros(tf.shape(neutral)), pos_neg[:, 1:]], 1,
										 name="trainscores")
		# Calculate Mean cross-entropy loss
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.training_scores, labels=self.input_y)
			self.training_loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		# Accuracy
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "accuracy")

			self.training_predictions = tf.argmax(self.training_scores, 1, name="predictions")
			correct_predictions = tf.equal(self.training_predictions, tf.argmax(self.input_y, 1))
			self.training_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name = "trainaccuracy")
