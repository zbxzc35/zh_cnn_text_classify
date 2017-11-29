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
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

tf.flags.DEFINE_string("data_dir", "./data/processed/training/", "Data source for classification.")
tf.flags.DEFINE_string("dev_dir", "./data/processed/validation/", "Data source for classification.")
tf.flags.DEFINE_string("test_dir", "./data/processed/testing/", "Data source for classification.")

tf.flags.DEFINE_integer("num_labels", None, "Number of labels for data. (default: None)")
tf.flags.DEFINE_integer("max_document_len", 500, "Max document lenth. (default: None)")

tf.flags.DEFINE_boolean("word_segment", False, "Whether do word segmentation. (default: False)")

tf.flags.DEFINE_string("wordembedding_name", "trained_word2vec.model.all", "Word embedding model name. (default: trained_word2vec.model)")

# Model hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 512, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-spearated filter sizes (default: '2,3,4,5')")
tf.flags.DEFINE_integer("num_filters", 256, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("reverse_grad_lambda", 0.1, "reverse gradient regularization lambda (default: 1.0)")

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
x, y, d = data_helpers.load_positive_negative_data_files(FLAGS)
x_target, y_target = data_helpers.load_dev_data_files(FLAGS.dev_dir)



# Get embedding vector
x, max_document_length = data_helpers.padding_sentences(x, '<PADDING>',word_segment= FLAGS.word_segment,
                                                                padding_sentence_length=FLAGS.max_document_len)

# Get embedding vector
x_target, _ = data_helpers.padding_sentences(x_target, '<PADDING>',word_segment= FLAGS.word_segment,
                                                                padding_sentence_length=FLAGS.max_document_len)

x_test, _ = data_helpers.load_dev_data_files(FLAGS.test_dir)

if not os.path.exists(_w2v_path):
    _, w2vModel = word2vec_helpers.embedding_sentences(sentences = x + x_target + x_test,
                                                       embedding_size = FLAGS.embedding_dim, file_to_save = _w2v_path)
else:
    _, w2vModel = word2vec_helpers.embedding_sentences(sentences = None ,
                                                       embedding_size = FLAGS.embedding_dim, file_to_load = _w2v_path)


FLAGS.embedding_dim = w2vModel.vector_size
print ('wordembedding.dim = {}'.format(FLAGS.embedding_dim))
print ('wordembedding.lenth = {}'.format(len(w2vModel.wv.vocab)))

x = np.array(x)
x_target = np.array(x_target)
print("x.shape = {}".format(x.shape))
print("y.shape = {}".format(y.shape))
print("d.shape = {}".format(d.shape))

# Save params
training_params_file = os.path.join(out_dir, 'training_params.pickle')
params = {'num_labels' : FLAGS.num_labels, 'max_document_length' : max_document_length}
data_helpers.saveDict(params, training_params_file)

# Shuffle data randomly
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_train = x[shuffle_indices]
y_train = y[shuffle_indices]

# TODO: This is very crude, should use cross-validation

shuffle_indices = np.random.permutation(np.arange(len(y_target)))
x_target = x_target[shuffle_indices]
y_target = y_target[shuffle_indices]

dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_target)))
x_t, x_dev = x_target[:dev_sample_index], x_target[dev_sample_index:]
y_t, y_dev = y_target[:dev_sample_index], y_target[dev_sample_index:]


print("Train/Target/Dev split: {:d}/{:d}/{:d}".format(len(y_train), len(y_t), len(y_dev)))

# Training
# =======================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement = FLAGS.allow_soft_placement,
        log_device_placement = FLAGS.log_device_placement)
    sess = tf.Session(config = session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length = x_train.shape[1],
            num_classes = y_train.shape[1],
            num_labels= d.shape[1],
            embedding_size = FLAGS.embedding_dim,
            filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters = FLAGS.num_filters,
            l2_reg_lambda = FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(cnn.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        grads_domain = optimizer.compute_gradients(cnn.d_loss)
        domain_op = optimizer.apply_gradients(grads_domain)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
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
        d_loss_summary = tf.summary.scalar("d_loss", cnn.d_loss)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, d_loss_summary, grad_summaries_merged])
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

        def train_step(x_batch, y_batch, d_batch, p):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.d_label: d_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                cnn.reverse_grad_lambda: 2. / (1. + np.exp(-10. * p)) - 1,
                cnn.learning_rate : 0.002 / (1. + 10 * p)**0.75
            }

            _,_, step, summaries, loss, accuracy, d_loss = sess.run(
                [train_op, domain_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.d_loss],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: label step {}, loss {:g}, acc {:g}, d_loss {:g}".format(time_str, step, loss, accuracy, d_loss))
            train_summary_writer.add_summary(summaries, step)

        def target_step(x_batch, y_batch,p = 1e-3):
            """
            Evaluates model on a dev set
            """

            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                cnn.learning_rate: 0.002 / (1. + 10 * p)**0.75
            }
            _, step, summaries, loss, accuracy, predictions = sess.run(
                [train_op,global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: train step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            print(x_batch.shape)
            print(y_batch.shape)
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, predictions = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            print([x.argmax() for x in y_batch])
            print([x for x in predictions])
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(list(zip(x_train, y_train,d)), FLAGS.batch_size, FLAGS.num_epochs)


        # Training loop. For each batch...
        for (s,batch) in enumerate(batches):

            x_batch, y_batch, d_batch = zip(*batch)
            x_batch_embedding, _ = word2vec_helpers.embedding_sentences(sentences= x_batch, embedding_size = FLAGS.embedding_dim,
                                                                        file_to_load = _w2v_path, model=w2vModel)
            x_batch = np.array(x_batch_embedding)
            p = float(s) / (float(len(x_train)*FLAGS.num_epochs)/FLAGS.batch_size)
            train_step(x_batch, y_batch, d_batch, p)

            shuffle_indices = np.random.permutation(np.arange(len(x_t)))[:int(FLAGS.batch_size/4)]

            x_t_batch = x_t[shuffle_indices]
            y_t_batch = y_t[shuffle_indices]

            x_batch_embedding, _ = word2vec_helpers.embedding_sentences(sentences= x_t_batch, embedding_size = FLAGS.embedding_dim,
                                                                        file_to_load = _w2v_path, model=w2vModel)
            x_t_batch = np.array(x_batch_embedding)
            target_step(x_t_batch, y_t_batch,p)

            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                shuffle_indices = np.random.permutation(np.arange(len(x_dev)))[:FLAGS.batch_size*4]
                x_dev_shuffled = x_dev[shuffle_indices]
                y_dev_shuffled = y_dev[shuffle_indices]

                x_batch_embedding, _ = word2vec_helpers.embedding_sentences(sentences= x_dev_shuffled, embedding_size=FLAGS.embedding_dim,
                                                                            file_to_load=_w2v_path, model=w2vModel)
                x_dev_shuffled = np.array(x_batch_embedding)
                print("\nEvaluation:")
                dev_step(x_dev_shuffled, y_dev_shuffled, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
