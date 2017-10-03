# encoding: UTF-8

import numpy as np
import re
import itertools
from collections import Counter
import os
import word2vec_helpers
import time
import pickle
from os import listdir
from os.path import isfile, join

def load_data_and_labels(input_text_file, input_label_file, num_labels):
    x_text = read_and_clean_zh_file(input_text_file)
    y = None if not os.path.exists(input_label_file) else map(int, list(open(input_label_file, "r").readlines()))
    return (x_text, y)

def load_positive_negative_data_files(FLAGS):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    # Ads_Marketing_examples         = read_and_clean_zh_file(FLAGS.Ads_Marketing_file)
    # Agent_Issues_examples          = read_and_clean_zh_file(FLAGS.Agent_Issues_file)
    # Charity_Events_examples        = read_and_clean_zh_file(FLAGS.Charity_Events_file)
    # Contact_Information_examples   = read_and_clean_zh_file(FLAGS.Contact_Information_file)
    # Corporate_Brands_examples      = read_and_clean_zh_file(FLAGS.Corporate_Brands_file)
    # Corporate_News_examples        = read_and_clean_zh_file(FLAGS.Corporate_News_file)
    # Customer_Service_examples      = read_and_clean_zh_file(FLAGS.Customer_Service_file)
    # Employment_examples            = read_and_clean_zh_file(FLAGS.Employment_file)
    # Fund_examples                  = read_and_clean_zh_file(FLAGS.Fund_file)
    # Health_Information_examples    = read_and_clean_zh_file(FLAGS.Health_Information_file)
    # Irrelevant_Ads_examples        = read_and_clean_zh_file(FLAGS.Irrelevant_Ads_file)
    # Life_Comprehend_examples       = read_and_clean_zh_file(FLAGS.Life_Comprehend_file)
    # Products_examples              = read_and_clean_zh_file(FLAGS.Products_file)
    # Products_Service_examples      = read_and_clean_zh_file(FLAGS.Products_Service_file)
    # Recruitment_examples           = read_and_clean_zh_file(FLAGS.Recruitment_file)
    # Sponsored_Events_examples      = read_and_clean_zh_file(FLAGS.Sponsored_Events_file)
    # Survey_Questions_examples      = read_and_clean_zh_file(FLAGS.Survey_Questions_file)
    # Volunteering_Activity_examples = read_and_clean_zh_file(FLAGS.Volunteering_Activity_file)
    # Website_Issues_examples        = read_and_clean_zh_file(FLAGS.Website_Issues_file)
    #
    # General_Mentioned_examples     = read_and_clean_zh_file(FLAGS.General_Mentioned_file)
    # Stocks_Earnings_examples       = read_and_clean_zh_file(FLAGS.Stocks_Earnings_file)
    mypath = FLAGS.data_dir
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    FLAGS.num_labels = len(onlyfiles)

    # Combine data
    examples = []
    for i in range(len(onlyfiles)):
        examples.append(read_and_clean_zh_file(mypath + onlyfiles[i]))
        if i == 0:
            x_text = examples[i][:]
        else:
            x_text += examples[i][:]




    # Generate labels
    I = np.eye(len(onlyfiles), dtype=int)
    for i in range(len(onlyfiles)):
        if i == 0:
            y = [I[i] for _ in examples[i]]
        else:
            y += [I[i] for _ in examples[i]]


    # y = np.concatenate([Ads_Marketing_labels, Agent_Issues_labels, Charity_Events_labels,
    #                     Contact_Information_labels, Corporate_Brand_labels, Corporate_News_labels,
    #                     Customer_Service_labels, Employment_labels, Fund_labels, Health_Information_labels,
    #                     Irrelevant_Ads_labels, Life_Comprehend_labels, Products_labels, Products_Service_labels,
    #                     Recruitment_labels, Sponsored_Events_labels, Survey_Questions_labels,
    #                     Volunteering_Activity_labels, Website_Issues_labels, General_Mentioned_labels, Stocks_Earnings_labels], 0)
    return [x_text, np.array(y)]

def padding_sentences(input_sentences, padding_token, padding_sentence_length = None):
    sentences = [sentence.split(' ') for sentence in input_sentences]
    max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in sentences])
    for sentence in sentences:
        if len(sentence) > max_sentence_length:
            sentence = sentence[:max_sentence_length]
        else:
            sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
    return (sentences, max_sentence_length)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    '''
    Generate a batch iterator for a dataset
    '''
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
	    # Shuffle the data at each epoch
	    shuffle_indices = np.random.permutation(np.arange(data_size))
	    shuffled_data = data[shuffle_indices]
	else:
	    shuffled_data = data
	for batch_num in range(num_batches_per_epoch):
	    start_idx = batch_num * batch_size
	    end_idx = min((batch_num + 1) * batch_size, data_size)
	    yield shuffled_data[start_idx : end_idx]

def test():
    # Test clean_str
    print("Test")
    #print(clean_str("This's a huge dog! Who're going to the top."))
    # Test load_positive_negative_data_files
    #x_text,y = load_positive_negative_data_files("./tiny_data/rt-polarity.pos", "./tiny_data/rt-polarity.neg")
    #print(x_text)
    #print(y)
    # Test batch_iter
    #batches = batch_iter(x_text, 2, 4)
    #for batch in batches:
    #    print(batch)

def mkdir_if_not_exist(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

def seperate_line(line):
    return ''.join([word + ' ' for word in line])

def read_and_clean_zh_file(input_file, output_cleaned_file = None):
    lines = list(open(input_file, "r").readlines())
    lines = [clean_str(seperate_line(line.decode('utf-8'))) for line in lines]
    if output_cleaned_file is not None:
        with open(output_cleaned_file, 'w') as f:
            for line in lines:
                f.write((line + '\n').encode('utf-8'))
    return lines

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(ur"[^\u4e00-\u9fff]", " ", string)
    #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    #string = re.sub(r"\'s", " \'s", string)
    #string = re.sub(r"\'ve", " \'ve", string)
    #string = re.sub(r"n\'t", " n\'t", string)
    #string = re.sub(r"\'re", " \'re", string)
    #string = re.sub(r"\'d", " \'d", string)
    #string = re.sub(r"\'ll", " \'ll", string)
    #string = re.sub(r",", " , ", string)
    #string = re.sub(r"!", " ! ", string)
    #string = re.sub(r"\(", " \( ", string)
    #string = re.sub(r"\)", " \) ", string)
    #string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    #return string.strip().lower()
    return string.strip()

def saveDict(input_dict, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(input_dict, f) 

def loadDict(dict_file):
    output_dict = None
    with open(dict_file, 'rb') as f:
        output_dict = pickle.load(f)
    return output_dict
