# -*- coding: utf-8 -*-
import os

import numpy as np
import tensorflow as tf

import DataSet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# reset graph
tf.reset_default_graph()


# weight init function
def weight_variable(shape):
    initial = (np.random.randn(shape[0], shape[1])) / np.sqrt(shape[0] / 2)
    return tf.Variable(initial, dtype=tf.float32)


# bias init function
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# cnn weight init function
def weight_variable_cnn(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# cnn bias init function
def bias_variable_cnn(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 2D cnn layer init function
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# max pooling layer with kernal = [1,2,2,1]
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# one of concat measure: full matching
def getFullMatching(seq, state, W):  # （所有隐层、最后输出、w）
    tiledSeq = seq
    tiledState = tf.tile(tf.expand_dims(state, 1), [1, 100, 1])

    result = []
    for i in range(20):
        weightedTiledSeq = tf.multiply(tiledSeq, W[i])
        weightedTiledState = tf.multiply(tiledState, W[i])

        weightedTiledSeqNorm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(weightedTiledSeq), axis=-1), 1e-6))
        weightedTiledStateNorm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(weightedTiledState), axis=-1), 1e-6))

        norm = tf.multiply(weightedTiledSeqNorm, weightedTiledStateNorm)

        fullMatching = tf.div(tf.reduce_sum(tf.multiply(weightedTiledSeq, weightedTiledState), -1), norm)

        result.append(fullMatching)
    ans = tf.reshape(tf.concat(result, -1), [-1, 100, 20])

    return ans


'''
0. init layer
'''
dataset = DataSet.DataSet()

BATCH_SIZE = 1024
MAX_SEQUENCE_LENGTH = 100
MAX_TITLE_LENGTH = 5
LSTM_HIDDEN = 100
EMBEDDING_DIM = len(dataset.zh_embedding_matrix[0])
ZH_EMBEDDING_VOCABULARY_SIZE = len(dataset.zh_embedding_matrix)
EN_EMBEDDING_VOCABULARY_SIZE = len(dataset.en_embedding_matrix)

# placeholder
title_ids_1 = tf.placeholder(tf.int32, shape=[None, MAX_TITLE_LENGTH])
title_ids_2 = tf.placeholder(tf.int32, shape=[None, MAX_TITLE_LENGTH])
title_ids_3 = tf.placeholder(tf.int32, shape=[None, MAX_TITLE_LENGTH])

word_ids_1 = tf.placeholder(tf.int32, shape=[None, MAX_SEQUENCE_LENGTH])
word_ids_2 = tf.placeholder(tf.int32, shape=[None, MAX_SEQUENCE_LENGTH])
word_ids_3 = tf.placeholder(tf.int32, shape=[None, MAX_SEQUENCE_LENGTH])

sequence_lengths_1 = tf.placeholder(tf.int32, shape=[None])
sequence_lengths_2 = tf.placeholder(tf.int32, shape=[None])
sequence_lengths_3 = tf.placeholder(tf.int32, shape=[None])

L1 = tf.placeholder(shape=[ZH_EMBEDDING_VOCABULARY_SIZE, EMBEDDING_DIM], dtype=tf.float32)
L2 = tf.placeholder(shape=[EN_EMBEDDING_VOCABULARY_SIZE, EMBEDDING_DIM], dtype=tf.float32)

keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)
regularization_rate = tf.placeholder(tf.float32)
margin = tf.placeholder(tf.float32)

'''
1. embedding layer
'''
pretrained_embeddings_1 = tf.nn.embedding_lookup(L1, word_ids_1)
pretrained_embeddings_2 = tf.nn.embedding_lookup(L2, word_ids_2)
pretrained_embeddings_3 = tf.nn.embedding_lookup(L2, word_ids_3)

pretrained_title_embedding_1 = tf.nn.embedding_lookup(L1, title_ids_1)
pretrained_title_embedding_2 = tf.nn.embedding_lookup(L2, title_ids_2)
pretrained_title_embedding_3 = tf.nn.embedding_lookup(L2, title_ids_3)

pretrained_title_1 = tf.reduce_mean(pretrained_title_embedding_1, 1)
pretrained_title_2 = tf.reduce_mean(pretrained_title_embedding_2, 1)
pretrained_title_3 = tf.reduce_mean(pretrained_title_embedding_3, 1)

'''
2. lstm layer
'''
lstm1_fw_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_HIDDEN)
lstm1_bw_cell = tf.contrib.rnn.BasicLSTMCell(LSTM_HIDDEN)

with tf.variable_scope("Bi-lstm1", reuse=None, initializer=tf.orthogonal_initializer()) as scope:
    ((bilstm1_fw_outputs1,
      bilstm1_bw_outputs1),
     (bilstm1_fw_final_state1,
      bilstm1_bw_final_state1)) = tf.nn.bidirectional_dynamic_rnn(lstm1_fw_cell, lstm1_bw_cell,
                                                                  pretrained_embeddings_1,
                                                                  sequence_length=sequence_lengths_1,
                                                                  dtype=tf.float32)
    bilstm1_state_h1 = tf.concat((bilstm1_fw_final_state1.h, bilstm1_bw_final_state1.h), 1)

with tf.variable_scope("Bi-lstm1", reuse=True) as scope:
    ((bilstm1_fw_outputs2,
      bilstm1_bw_outputs2),
     (bilstm1_fw_final_state2,
      bilstm1_bw_final_state2)) = tf.nn.bidirectional_dynamic_rnn(lstm1_fw_cell, lstm1_bw_cell,
                                                                  pretrained_embeddings_2,
                                                                  sequence_length=sequence_lengths_2,
                                                                  dtype=tf.float32)
    bilstm1_state_h2 = tf.concat((bilstm1_fw_final_state2.h, bilstm1_bw_final_state2.h), 1)

with tf.variable_scope("Bi-lstm1", reuse=True) as scope:
    ((bilstm1_fw_outputs3,
      bilstm1_bw_outputs3),
     (bilstm1_fw_final_state3,
      bilstm1_bw_final_state3)) = tf.nn.bidirectional_dynamic_rnn(lstm1_fw_cell, lstm1_bw_cell,
                                                                  pretrained_embeddings_3,
                                                                  sequence_length=sequence_lengths_3,
                                                                  dtype=tf.float32)
    bilstm1_state_h3 = tf.concat((bilstm1_fw_final_state3.h, bilstm1_bw_final_state3.h), 1)

encoder_final_output3 = tf.nn.dropout(tf.concat((bilstm1_fw_outputs3, bilstm1_bw_outputs3), -1), keep_prob)
encoder_final_output2 = tf.nn.dropout(tf.concat((bilstm1_fw_outputs2, bilstm1_bw_outputs2), -1), keep_prob)
encoder_final_output1 = tf.nn.dropout(tf.concat((bilstm1_fw_outputs1, bilstm1_bw_outputs1), -1), keep_prob)

'''
3. match layer
'''
W_matching = weight_variable([20, 200])

FullMatching12 = getFullMatching(encoder_final_output1, bilstm1_state_h2, W_matching)
FullMatching13 = getFullMatching(encoder_final_output1, bilstm1_state_h3, W_matching)
FullMatching21 = getFullMatching(encoder_final_output2, bilstm1_state_h1, W_matching)
FullMatching31 = getFullMatching(encoder_final_output3, bilstm1_state_h1, W_matching)

zh_pos_match = tf.concat([FullMatching12, FullMatching21], -1)
zh_neg_match = tf.concat([FullMatching13, FullMatching31], -1)

'''
4.CNN layer
'''
zh_pos_match = tf.reshape(zh_pos_match, [-1, 100, 40, 1])  # tf.expand_dims(zh_pos_match,-1)
zh_neg_match = tf.reshape(zh_neg_match, [-1, 100, 40, 1])  # tf.expand_dims(zh_neg_match,-1)

# 第一层卷积
W_conv1 = weight_variable_cnn([5, 5, 1, 32])
b_conv1 = bias_variable_cnn([32])

h_conv1_pos = tf.nn.relu(conv2d(zh_pos_match, W_conv1) + b_conv1)
h_pool1_pos = max_pool_2x2(h_conv1_pos)

h_conv1_neg = tf.nn.relu(conv2d(zh_neg_match, W_conv1) + b_conv1)
h_pool1_neg = max_pool_2x2(h_conv1_neg)

# 第二层卷积
W_conv2 = weight_variable_cnn([5, 5, 32, 64])
b_conv2 = bias_variable_cnn([64])

h_conv2_pos = tf.nn.relu(conv2d(h_pool1_pos, W_conv2) + b_conv2)
h_pool2_pos = max_pool_2x2(h_conv2_pos)

h_conv2_neg = tf.nn.relu(conv2d(h_pool1_neg, W_conv2) + b_conv2)
h_pool2_neg = max_pool_2x2(h_conv2_neg)

pos = tf.reshape(h_pool2_pos, [-1, 25 * 10 * 64])
neg = tf.reshape(h_pool2_neg, [-1, 25 * 10 * 64])

'''
5. score
'''
W_fc1 = weight_variable([25 * 10 * 64, 1024])
b_fc1 = bias_variable([1024])

with tf.variable_scope("fc1", reuse=None) as scope:
    hid_pos1 = tf.nn.relu(tf.matmul(pos, W_fc1) + b_fc1)
with tf.variable_scope("fc1", reuse=True) as scope:
    hid_neg1 = tf.nn.relu(tf.matmul(neg, W_fc1) + b_fc1)

W_fc2 = weight_variable([1024, 100])
b_fc2 = bias_variable([100])

with tf.variable_scope("fc2", reuse=None) as scope:
    hid_pos2 = tf.nn.relu(tf.matmul(hid_pos1, W_fc2) + b_fc2)
with tf.variable_scope("fc2", reuse=True) as scope:
    hid_neg2 = tf.nn.relu(tf.matmul(hid_neg1, W_fc2) + b_fc2)

W_fc3 = weight_variable([100, 1])

with tf.variable_scope("fc3", reuse=None) as scope:
    output_pos = tf.matmul(hid_pos2, W_fc3)
with tf.variable_scope("fc3", reuse=True) as scope:
    output_neg = tf.matmul(hid_neg2, W_fc3)

abs_feature_score1 = tf.reshape(output_pos, [-1])
abs_feature_score2 = tf.reshape(output_neg, [-1])

abs_loss = tf.reduce_mean(tf.maximum(margin + abs_feature_score2 - abs_feature_score1, 0.), -1)

tv = tf.trainable_variables()
regularization_cost = regularization_rate * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

score1 = abs_feature_score1
score2 = abs_feature_score2

'''
6. loss, accuracy, and optimizer
'''
loss = abs_loss + regularization_cost
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
acc = tf.reduce_mean(tf.maximum(tf.sign(score1 - score2), 0.))

'''
7. run
'''
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.Saver()
sess.run(init)

'''
8. train
'''
max_prob = 0.0
val_batch_data, val_abs_data = dataset.get_test_data(128)[:500]  # 验证集
f = open('log_train.txt', 'w')

for i in range(30000):
    batch = dataset.get_train_batch(128)  # 训练集

    sess.run(optimizer, feed_dict={
        word_ids_1: batch[5][0],
        word_ids_2: batch[5][1],
        word_ids_3: batch[5][2],
        sequence_lengths_1: batch[5][3],
        sequence_lengths_2: batch[5][4],
        sequence_lengths_3: batch[5][5],
        title_ids_1: batch[5][6],
        title_ids_2: batch[5][7],
        title_ids_3: batch[5][8],

        L1: dataset.zh_embedding_matrix, L2: dataset.en_embedding_matrix,

        keep_prob: 0.5, learning_rate: 0.001, regularization_rate: 0.0001, margin: 1.0})

    if i % 100 == 0:
        loss_train_batch, acc_train_batch = sess.run([loss, acc],
                                                     feed_dict={
                                                         word_ids_1: batch[5][0],
                                                         word_ids_2: batch[5][1],
                                                         word_ids_3: batch[5][2],
                                                         sequence_lengths_1: batch[5][3],
                                                         sequence_lengths_2: batch[5][4],
                                                         sequence_lengths_3: batch[5][5],
                                                         title_ids_1: batch[5][6],
                                                         title_ids_2: batch[5][7],
                                                         title_ids_3: batch[5][8],
                                                         L1: dataset.zh_embedding_matrix,
                                                         L2: dataset.en_embedding_matrix,
                                                         keep_prob: 1.0, 
                                                         learning_rate: 0.0001,
                                                         regularization_rate: 0.0001, 
                                                         margin: 1.0})
        f.write("Iter " + str(i) + ":train batch loss:" + "{:.6f}".format(
            loss_train_batch) + ":train batch acc:" + "{:.6f}".format(acc_train_batch) + '\n')
        print("Iter " + str(i) + ":train batch loss:" + "{:.6f}".format(
            loss_train_batch) + ":train batch acc:" + "{:.6f}".format(acc_train_batch))
    if i % 500 == 0:
        acc1 = 0.
        acc10 = 0.

        for j in range(len(val_batch_data)):
            score = sess.run(score1, feed_dict={
                word_ids_1: val_abs_data[j][0],
                word_ids_2: val_abs_data[j][2],
                sequence_lengths_1: val_abs_data[j][1],
                sequence_lengths_2: val_abs_data[j][3],
                title_ids_1: val_abs_data[j][4],
                title_ids_2: val_abs_data[j][5],
                L1: dataset.zh_embedding_matrix, 
                L2: dataset.en_embedding_matrix,
                keep_prob: 1.0, 
                learning_rate: 0.0001, 
                regularization_rate: 0.0001, 
                margin: 1.0})
            ttl = score.tolist()
            assert len(ttl) == 100
            xxl = ttl[0]
            n = 0
            for j in range(1, len(ttl)):
                if xxl <= ttl[j]:  # score越大越好
                    n += 1
            if n == 0:
                acc1 += 1
            if n < 10:
                acc10 += 1
        acc1 = acc1 / len(val_batch_data)
        acc10 = acc10 / len(val_batch_data)
        f.write("Iter " + str(i) + ":val top 1 acc:" + "{:.6f}".format(acc1) + " val top 10 acc:" + "{:.6f}".format(
            acc10) + '\n\n')
        print("Iter " + str(i) + ":val top 1 acc:" + "{:.6f}".format(acc1) + " val top 10 acc:" + "{:.6f}".format(
            acc10))

        if acc1 > max_prob:
            f.write("save the acc" + str(acc1) + '+++++++++++++++++++++++++++++++++++\n')
            print ("save the acc" + str(acc1))
            max_prob = acc1
            saver.save(sess, "./BiMPM-cnn.ckpt")

'''
9. test
'''
test_batch_data, test_abs_data = dataset.get_test_data()

acc1 = 0.
acc10 = 0.
saver.restore(sess, "./BiMPM-cnn.ckpt")
f = open('log_test_bimpm.txt', 'w')
for j in range(len(test_batch_data)):
    f.write('item:' + str(j) + '\n')
    score = sess.run(score1, feed_dict={
        word_ids_1: test_abs_data[j][0],
        word_ids_2: test_abs_data[j][2],
        sequence_lengths_1: test_abs_data[j][1],
        sequence_lengths_2: test_abs_data[j][3],
        title_ids_1: test_abs_data[j][4],
        title_ids_2: test_abs_data[j][5],
        L1: dataset.zh_embedding_matrix,
        L2: dataset.en_embedding_matrix,
        keep_prob: 1.0,
        learning_rate: 0.0001,
        regularization_rate: 0.0001,
        margin: 1.0})

    n = 0
    for k in range(1, 100):
        if score[0] <= score[k]:
            f.write(str(score[k]) + '\n')
            n += 1
    if n == 0:
        acc1 += 1
    if n < 10:
        acc10 += 1
    f.write('\n')
    f.write(str(acc1) + '\n')
    f.write(str(acc10) + '\n')
    f.write('\n')

acc1 = acc1 / len(test_batch_data)
acc10 = acc10 / len(test_batch_data)
f.write("test top 1 acc:" + "{:.6f}".format(acc1) + " test top 10 acc:" + "{:.6f}".format(acc10) + '\n')
f.close()
