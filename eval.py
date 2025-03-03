#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import data_helpers
import word2vec_helpers
from text_cnn import TextCNN
import csv


# Parameters
# ==================================================
#all 24 catigories :2017-10-05T07:04:31.212577,model.512
#2 catigories 98.3: 2017-10-14T07:06:27.827187
#2 catigories 98.4: 2017-10-14T13:26:17.814401
# Data Parameters
tf.flags.DEFINE_string("data_dir", "./data/processed/testing/", "Test text data source to evaluate.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/2017-10-16T20:26:22.272819/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")
tf.flags.DEFINE_string("wordembedding_name", "trained_word2vec.model.512", "Word embedding model name. (default: trained_word2vec.model)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# validate
# ==================================================

# validate checkout point file
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
if checkpoint_file is None:
    print("Cannot find a valid checkpoint file!")
    exit(0)
print("Using checkpoint file : {}".format(checkpoint_file))

# validate word2vec model file
trained_word2vec_model_file = os.path.join("./runs/", FLAGS.wordembedding_name)
if not os.path.exists(trained_word2vec_model_file):
    print("Word2vec model file \'{}\' doesn't exist!".format(trained_word2vec_model_file))
print("Using word2vec model file : {}".format(trained_word2vec_model_file))

# validate training params file
training_params_file = os.path.join(FLAGS.checkpoint_dir, "..", "training_params.pickle")
if not os.path.exists(training_params_file):
    print("Training params file \'{}\' is missing!".format(training_params_file))
print("Using training params file : {}".format(training_params_file))

# Load params
params = data_helpers.loadDict(training_params_file)
num_labels = int(params['num_labels'])
max_document_length = int(params['max_document_length'])

# Load data
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_positive_negative_data_files(FLAGS)
else:
    x_raw = ["a masterpiece four years in the making", "everything is off."]
    y_test = [1, 0]

# Get Embedding vector x_test
print max_document_length
x_test, max_document_length = data_helpers.padding_sentences(x_raw, '<PADDING>', padding_sentence_length = max_document_length)
_, w2vModel = word2vec_helpers.embedding_sentences(file_to_load = trained_word2vec_model_file)
x_test = np.array(x_test)

print("x_test.shape = {}".format(x_test.shape))


# Evaluation
# ==================================================
print("\nEvaluating...\n")
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        with tf.device('/cpu:0'):
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for x_test_batch in batches:
                x_batch_embedding, _ = word2vec_helpers.embedding_sentences(sentences=x_test_batch, model=w2vModel)
                x_test_batch = np.array(x_batch_embedding)
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    onlyfiles = [f for f in listdir(FLAGS.data_dir) if isfile(join(FLAGS.data_dir, f))]
    lable_dict = {i: word.split('.')[0] for i, word in enumerate(onlyfiles)}
    print lable_dict
    y_vect = np.array([lable_dict[x.argmax()] for x in y_test])
    all_predictions = np.array([lable_dict[int(x)] for x in all_predictions])
    correct_predictions = float(np.sum(all_predictions == y_vect))

    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.asarray([np.array([text.encode('utf-8') for text in x_raw]), y_vect, all_predictions])
print predictions_human_readable.shape
out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'w') as f:
    csv.writer(f).writerows(np.transpose(predictions_human_readable))

