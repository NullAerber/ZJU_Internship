#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import random
import re
import time
from datetime import timedelta

import keras
import numpy as np
import tensorflow as tf
from gensim.models import word2vec

from textcnn_model_v1 import TCNNConfig, TextCNN

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


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
        # text8 需要自己下载，Google训练embedding的训练集
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


def train(train_x, train_y, val_x, val_y):
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 创建session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(train_x, train_y, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)

            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, val_x, val_y)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


if __name__ == '__main__':
    print('Configuring CNN model...')
    # step 1: load data
    print("Loading training and validation data...")
    start_time = time.time()
    word2vec_bin_file = './data/word2vec_model.model'
    pos_file = 'data/rt-polarity.pos'
    neg_file = 'data/rt-polarity.neg'
    train_x, train_y, test_x, test_y, dict_vec = get_embedding(word2vec_bin_file, pos_file, neg_file)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # step 2: create CNN model
    config = TCNNConfig()
    config.embedding_dim = len(dict_vec[0])
    config.seq_length = len(train_x[0])
    config.vocab_size = len(dict_vec)
    config.num_classes = 2
    model = TextCNN(config)

    # step 3： start training
    train(train_x, train_y, test_x, test_y)
