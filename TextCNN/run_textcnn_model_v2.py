# -*- coding: utf-8 -*-
import re

import keras
import numpy as np
import os
from gensim.models import word2vec
import random

from textcnn_model_v2 import TextCNN


def clean_string(string, TREC=False):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def get_embedding(word2vec_bin_file, pos_file, neg_file, cv=10, clear_flag=True):
    # 训练集的语料词汇表和在训练集上词的词频
    dict_cab = set([])
    # 纯文本
    raw_review = []
    max_length = 0
    '''
    step 1: get the total vocabulary set and raw review string from training set
    '''
    with open(pos_file, 'rb') as pos_f:
        for line in pos_f:
            review = []
            review.append(line.decode(encoding='ISO-8859-1').strip())
            if clear_flag:
                review = clean_string(" ".join(review))
            else:
                review = " ".join(review).lower()
            processed_review = {"y": 1,
                                "text": review,
                                "num_words": len(review.split()),
                                "spilt": np.random.randint(0, cv)}
            # processed_review = [review,1]
            raw_review.append(processed_review)
            temp_cab = set(review.split())
            dict_cab = dict_cab | temp_cab

            if max_length < processed_review['num_words']:
                max_length = processed_review['num_words']

    with open(neg_file, 'rb') as neg_f:
        for line in neg_f:
            review = []
            review.append(line.decode(encoding='ISO-8859-1').strip())
            if clear_flag:
                review = clean_string(" ".join(review))
            else:
                review = " ".join(review).lower()
            processed_review = {"y": 0,
                                "text": review,
                                "num_words": len(review.split()),
                                "spilt": np.random.randint(0, cv)}
            # processed_review = [review, 0]
            raw_review.append(processed_review)
            temp_cab = set(review.split())
            dict_cab = dict_cab | temp_cab

            if max_length < processed_review['num_words']:
                max_length = processed_review['num_words']

    '''
    step 2: load word embedding vector model , and if not exist, will train 
    '''
    if not os.path.exists(word2vec_bin_file):  # 如果不存在词汇表，重建
        sentences = word2vec.Text8Corpus('./data/text8')
        model = word2vec.Word2Vec(sentences, min_count=1)
        model.save(word2vec_bin_file)
    word2vec_model = word2vec.Word2Vec.load(word2vec_bin_file)

    '''
    step 3 : get word dict vector in training set
    '''
    embedding_size = word2vec_model['the'].__len__()
    dict_id = dict(zip(dict_cab, range(len(dict_cab))))
    dict_vec = np.zeros([len(dict_id), embedding_size])
    for word, id in dict_id.items():
        if word not in word2vec_model:
            dict_vec[id] = np.random.uniform(-0.25, 0.25, embedding_size)
        else:
            dict_vec[id] = word2vec_model[word]

    '''
    step 4 : trans the raw review to embeddding id
    '''
    random.shuffle(raw_review)
    x, y = [], []
    for review in raw_review:
        signal_vec = []
        for word in review['text'].split():
            signal_vec.append(dict_id[word])
        x.append(signal_vec)
        y.append(review['y'])
    x_pad = keras.preprocessing.sequence.pad_sequences(x, max_length)
    y_pad = keras.utils.to_categorical(y, num_classes=2)  # 将标签转换为one-hot表示

    '''
    step 5: split the data into traing data and test data
    '''
    train_x = x_pad[0:int(len(x) * 0.75)]
    train_y = y_pad[0:int(len(x) * 0.75)]
    test_x = x_pad[int(len(x) * 0.75) + 1:]
    test_y = y_pad[int(len(x) * 0.75) + 1:]

    return train_x, train_y, test_x, test_y, dict_vec


if __name__ == '__main__':
    # step 1: load data
    print("Loading training and validation data...")
    word2vec_bin_file = './data/word2vec_model.model'
    pos_file = 'data/rt-polarity.pos'
    neg_file = 'data/rt-polarity.neg'
    train_x, train_y, test_x, test_y, dict_vec = get_embedding(word2vec_bin_file, pos_file, neg_file)

    print('Configuring CNN model...')
    # step3 create TextCNN model
    text_cnn = TextCNN(dict_vec=dict_vec,
                       shuffer_falg=True,
                       static_falg=False,
                       num_classes=2,
                       filter_numbers=100,
                       filter_sizes=[3, 4, 5],
                       sentence_length=len(train_x[0]),
                       embedding_size=len(dict_vec[0]),
                       learnrate=1e-3,
                       epochs=10,
                       batch_size=64,
                       dropout_pro=0.5)
    # step4 start train
    text_cnn.train(train_x, train_y)
    # step5 validataion
    accur, loss = text_cnn.validataion(test_x, test_y)
    #
    print('accur is :{:.3f} loss is {:.3f}'.format(accur, loss))
    text_cnn.close()
