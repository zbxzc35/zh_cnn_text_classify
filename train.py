#! /usr/bin/env python
# encoding: utf-8

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import word2vec_helpers
from text_cnn import TextCNN

# Parameters
# =======================================================

# Data loading parameters
tf.flags.DEFINE_float("dev_sample_percentage", .15, "Percentage of the training data to use for validation")

tf.flags.DEFINE_string("data_dir", "./data/processed/", "Data source for classification.")

tf.flags.DEFINE_integer("num_labels", None, "Number of labels for data. (default: None)")
tf.flags.DEFINE_integer("max_document_len", 2500, "Max document lenth. (default: None)")

tf.flags.DEFINE_boolean("word_segment", False, "Whether do word segmentation. (default: False)")

tf.flags.DEFINE_string("wordembedding_name", "trained_word2vec.model", "Word embedding model name. (default: trained_word2vec.model)")

tf.flags.DEFINE_boolean("shuffle_data", True, "Whether or not shuffle the data. (default: False)")

tf.flags.DEFINE_string("positive_data_file", "Positive.utf8", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "Negative.utf8", "Data source for the negative data.")
tf.flags.DEFINE_string("neutral_data_file", "Neutral.utf8", "Data source for the neutral data.")

# Model hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 512, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-spearated filter sizes (default: '2,3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training paramters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evalue model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (defult: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Parse parameters from commands

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Prepare output directory for models and summaries
# =======================================================

timestamp = datetime.datetime.now().isoformat()
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
_w2v_path = os.path.join(os.path.curdir, "runs", FLAGS.wordembedding_name)
print("Writing to {}\n".format(out_dir))
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Data preprocess
# =======================================================

# Load data
print("Loading data...")
x_train, y_train,x_dev,y_dev = data_helpers.load_positive_negative_data_files(FLAGS)
# Get embedding vector
sentences, max_document_length = data_helpers.padding_sentences(np.concatenate([x_train,x_dev],axis=0), '<PADDING>',word_segment= FLAGS.word_segment,
                                                                padding_sentence_length=FLAGS.max_document_len)
if not os.path.exists(_w2v_path):
    _, w2vModel = word2vec_helpers.embedding_sentences(sentences = sentences,
                                                       embedding_size = FLAGS.embedding_dim, file_to_save = _w2v_path)
else:
    _, w2vModel = word2vec_helpers.embedding_sentences(sentences = None ,
                                                       embedding_size = FLAGS.embedding_dim, file_to_load = _w2v_path)

FLAGS.embedding_dim = w2vModel.vector_size
print ('wordembedding.dim = {}'.format(FLAGS.embedding_dim))
print ('wordembedding.lenth = {}'.format(len(w2vModel.wv.vocab)))

x = np.array(sentences)
x_train = x[:len(x_train)]
x_dev   = x[len(x_train):]
y = np.concatenate([y_train,y_dev],axis=0)
print("x.shape = {}".format(x_train.shape))
print("y.shape = {}".format(y_train.shape))

# Save params
training_params_file = os.path.join(out_dir, 'training_params.pickle')
params = {'num_labels' : FLAGS.num_labels, 'max_document_length' : max_document_length}
data_helpers.saveDict(params, training_params_file)

# Shuffle data randomly

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(x_train)))
x_train = x_train[shuffle_indices]
y_train = y_train[shuffle_indices]

shuffle_indices = np.random.permutation(np.arange(len(x_dev)))
x_shuffled = x_dev[shuffle_indices]
y_shuffled = y_dev[shuffle_indices]


# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_dev)))
x_dev, x_test = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_dev, y_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

x_dev = np.concatenate([x_dev, x_train[-5000:]])
y_dev = np.concatenate([y_dev, y_train[-5000:]])

x_train = x_train[:-5000]
y_train = y_train[:-5000]
print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev), len(y_test)))

_dev_per = 10

# Training
# =======================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement = FLAGS.allow_soft_placement,
        log_device_placement = FLAGS.log_device_placement)
    sess = tf.Session(config = session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length = x.shape[1],
            num_classes = y.shape[1],
            embedding_size = FLAGS.embedding_dim,
            filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters = FLAGS.num_filters,
            l2_reg_lambda = FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars_train = optimizer.compute_gradients(cnn.training_loss)
        grads_and_vars_dev = optimizer.compute_gradients(cnn.loss)
        print grads_and_vars_train
        print grads_and_vars_dev[-2]

        train_op = optimizer.apply_gradients(grads_and_vars_train[:-2], global_step=global_step)
        train_dev = optimizer.apply_gradients(grads_and_vars_dev[-2:], global_step=global_step)


        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars_train:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.training_loss, cnn.training_accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_dev,global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

            if writer:
                writer.add_summary(summaries, step)

        def test_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, prediction = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            print [elem.argmax() for elem in y_batch]
            print [elem for elem in prediction]

            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch_embedding, _ = word2vec_helpers.embedding_sentences(sentences= x_batch, embedding_size = FLAGS.embedding_dim,
                                                                        file_to_load = _w2v_path, model=w2vModel)
            x_batch = np.array(x_batch_embedding)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)

            #dev
            if current_step % _dev_per == 0:
                np.random.seed(None)
                shuffle_indices = np.random.permutation(np.arange(len(x_dev)))[:FLAGS.batch_size*4]
                x_dev_shuffled = x_dev[shuffle_indices]
                y_dev_shuffled = y_dev[shuffle_indices]

                x_batch_embedding, _ = word2vec_helpers.embedding_sentences(sentences= x_dev_shuffled, embedding_size=FLAGS.embedding_dim,
                                                                            file_to_load=_w2v_path, model=w2vModel)
                x_dev_batch = np.array(x_batch_embedding)
                print("\nEvaluation:")
                dev_step(x_dev_batch, y_dev_shuffled)
                print("")
                # test
                if current_step % FLAGS.evaluate_every == 0:
                    np.random.seed(None)
                    shuffle_indices = np.random.permutation(np.arange(len(x_test)))[:FLAGS.batch_size * 4]
                    x_test_shuffled = x_test[shuffle_indices]
                    y_test_shuffled = y_test[shuffle_indices]

                    x_batch_embedding, _ = word2vec_helpers.embedding_sentences(sentences=x_dev_shuffled,
                                                                                embedding_size=FLAGS.embedding_dim,
                                                                                file_to_load=_w2v_path, model=w2vModel)
                    x_test_batch = np.array(x_batch_embedding)
                    print("\nTesting:")
                    test_step(x_test_batch, y_test_shuffled, writer=dev_summary_writer)
                    print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
