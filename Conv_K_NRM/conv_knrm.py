# -*- coding: utf-8 -*-
import tensorflow as tf
import os
import DataSet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''
一些config数据
'''
dataset = DataSet.DataSet()
epochs = 5000
seq_length = 100
title_length = 5
filter_numbers = 128
batch_size = 256
embedding_dim = len(dataset.zh_embedding_matrix[0])
zh_vocab_size = len(dataset.zh_embedding_matrix)
en_vocab_size = len(dataset.en_embedding_matrix)
lstm_n_hidden = 64
lstm_n_layer = 3
cnn_layer_size_list = [3, 4, 5]
n_bins = 11
lamb = 0.5  # guassian_sigma = lamb * bin_size
regularization_rate = 0.15
hmax = 2

name = str(filter_numbers) + '_' + str(lstm_n_hidden) + '+' + str(cnn_layer_size_list) + '+' + str(
    batch_size) + '+conv_k_nrm'
name = name.replace('[', '').replace(']', '')

val_batch_data, val_abs_data = dataset.get_test_data()
test_batch_data, test_abs_data = dataset.get_test_data()


word_ids_zh = tf.placeholder(tf.int32, shape=[None, seq_length])
word_ids_pos = tf.placeholder(tf.int32, shape=[None, seq_length])
word_ids_neg = tf.placeholder(tf.int32, shape=[None, seq_length])

sequence_lengths_zh = tf.placeholder(tf.int32, shape=[None])
sequence_lengths_pos = tf.placeholder(tf.int32, shape=[None])
sequence_lengths_neg = tf.placeholder(tf.int32, shape=[None])

vob_zh_size = tf.placeholder(shape=[zh_vocab_size, embedding_dim], dtype=tf.float32)  # 改成variable
vob_en_size = tf.placeholder(shape=[en_vocab_size, embedding_dim], dtype=tf.float32)  # 改成variable

pretrained_embeddings_1 = tf.nn.embedding_lookup(vob_zh_size, word_ids_zh)
pretrained_embeddings_2 = tf.nn.embedding_lookup(vob_en_size, word_ids_pos)
pretrained_embeddings_3 = tf.nn.embedding_lookup(vob_en_size, word_ids_neg)

dropout_pro = tf.placeholder(tf.float32)
margin = tf.placeholder(tf.float32)
initial_learning_rate = tf.placeholder(tf.float32)

input_mu = tf.placeholder(tf.float32, shape=[n_bins], name='input_mu')
input_sigma = tf.placeholder(tf.float32, shape=[n_bins], name='input_sigma')

'''
初始化weight和bias
'''
def weight_variable(shape, name=None):
    # initial = tf.truncated_normal(shape, stddev=0.1,name=None)
    initial = tf.random_uniform(shape, -0.5, 0.5, name=None)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


'''
kernal
'''
def kernal_mus(n_kernels, use_exact):
    """
    get the mu for each guassian kernel. Mu is the middle of each bin
    :param n_kernels: number of kernels (including exact match). first one is exact match
    :return: l_mu, a list of mu.
    """
    if use_exact:
        l_mu = [1]
    else:
        l_mu = [2]
    if n_kernels == 1:
        return l_mu

    bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
    l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
    for i in xrange(1, n_kernels - 1):
        l_mu.append(l_mu[i] - bin_size)
    return l_mu


def kernel_sigmas(n_kernels, lamb, use_exact):
    """
    get sigmas for each guassian kernel.
    :param n_kernels: number of kernels (including exactmath.)
    :param lamb:
    :param use_exact:
    :return: l_sigma, a list of simga
    """
    bin_size = 2.0 / (n_kernels - 1)
    l_sigma = [0.00001]  # for exact match. small variance -> exact match
    if n_kernels == 1:
        return l_sigma

    l_sigma += [bin_size * lamb] * (n_kernels - 1)
    return l_sigma


'''
得到归一化的batch
'''
def get_batch_norm(conv, num):
    mean, var = tf.nn.moments(conv, axes=[0, 1, 2])  # 计算一阶矩（均值），以及二阶矩（方差）
    beta = tf.Variable(tf.zeros([num]), name="beta")
    gamma = weight_variable([num], name="gamma")
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True)
    return batch_norm


'''
Conv 卷积层
'''
def conv_layer(W, b, h, pretrained_embeddings_1, pretrained_embeddings_2, pretrained_embeddings_3, filter_numbers=128):
    # tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None)
    conv1 = tf.nn.conv2d(tf.expand_dims(pretrained_embeddings_1, -1), W, strides=[1, 1, embedding_dim, 1],
                         padding="SAME", name="conv1")
    # bn1=get_bn(conv1, filter_numbers)
    conv_output1 = tf.nn.dropout(tf.nn.bias_add(conv1, b, name="conv_output1"), dropout_pro)

    conv2 = tf.nn.conv2d(tf.expand_dims(pretrained_embeddings_2, -1), W, strides=[1, 1, embedding_dim, 1],
                         padding="SAME", name="conv2")
    # bn2=get_bn(conv2, filter_numbers)
    conv_output2 = tf.nn.dropout(tf.nn.bias_add(conv2, b, name="conv_output2"), dropout_pro)

    conv3 = tf.nn.conv2d(tf.expand_dims(pretrained_embeddings_3, -1), W, strides=[1, 1, embedding_dim, 1],
                         padding="SAME", name="conv3")
    # bn3=get_bn(conv3, filter_numbers)
    conv_output3 = tf.nn.dropout(tf.nn.bias_add(conv3, b, name="conv_output3"), dropout_pro)

    return conv_output1, conv_output2, conv_output3, W, b


'''
未进行优化的原始的knrm模型
'''
def original_knrm(pretrained_embeddings_1, pretrained_embeddings_2, mu, sigma):
    # normalize and compute similarity matrix
    norm_1 = tf.sqrt(tf.reduce_sum(tf.square(pretrained_embeddings_1), 2, keep_dims=True))
    normalized_1_embed = pretrained_embeddings_1 / (norm_1 + 0.0005)  # 分母可能为0或极小值
    norm_2 = tf.sqrt(tf.reduce_sum(tf.square(pretrained_embeddings_2), 2, keep_dims=True))
    normalized_2_embed = pretrained_embeddings_2 / (norm_2 + 0.0005)  #
    tmp = tf.transpose(normalized_2_embed, perm=[0, 2, 1])
    # similarity matrix [n_batch, qlen, dlen]
    sim = tf.matmul(normalized_1_embed, tmp, name='similarity_matrix')  # batch_matmul

    # compute gaussian kernel
    rs_sim = tf.reshape(sim, [-1, seq_length, seq_length, 1])  ####

    # compute Gaussian scores of each kernel
    tmp = tf.exp(-tf.square(tf.subtract(rs_sim, mu)) / (tf.multiply(tf.square(sigma), 2)))
    # sum up gaussian scores
    kde = tf.reduce_sum(tmp, [2])

    # kde = tf.log(tf.maximum(kde, 1e-10)) * 0.01  # 0.01 used to scale down the data.
    kde = tf.log(tf.clip_by_value(kde, 1e-10, 1e10)) * 0.01  ####how to deal with log=0

    # aggregated query terms
    aggregated_kde = tf.reduce_sum(kde, [1])

    feats = []  # store the soft-TF features from each field.
    feats.append(aggregated_kde)  # [[batch, nbins]]
    feats_tmp = tf.concat(feats, 1)  # [batch, n_bins]######

    # Reshape. (maybe not necessary...)
    feats_flat = tf.reshape(feats_tmp, [-1, n_bins])

    # return some mid result and final matching score.
    return feats_flat  # , o


'''
带有CNN的K NRM模型
'''
def conv_knrm_model(W1, b1, W_conv1, b_conv1, W_conv2, b_conv2, pretrained_embeddings_1, pretrained_embeddings_2,
                    pretrained_embeddings_3, mu, sigma, filter_numbers=128):
    conv_zh = []
    conv_en = []
    conv_ne = []
    conv_output1, conv_output2, conv_output3, W_conv1, b_conv1 = conv_layer(W_conv1, b_conv1, 1,
                                                                            pretrained_embeddings_1,
                                                                            pretrained_embeddings_2,
                                                                            pretrained_embeddings_3, filter_numbers)
    conv_zh.append(conv_output1)
    conv_en.append(conv_output2)
    conv_ne.append(conv_output3)

    conv_output1, conv_output2, conv_output3, W_conv2, b_conv2 = conv_layer(W_conv2, b_conv2, 2,
                                                                            pretrained_embeddings_1,
                                                                            pretrained_embeddings_2,
                                                                            pretrained_embeddings_3, filter_numbers)
    conv_zh.append(conv_output1)
    conv_en.append(conv_output2)
    conv_ne.append(conv_output3)

    feats1 = []
    feats2 = []
    for qi in range(2):
        for di in range(2):
            q_ngram = tf.reshape(conv_zh[qi], [-1, embedding_dim, filter_numbers])
            d_ngram = tf.reshape(conv_en[di], [-1, embedding_dim, filter_numbers])
            feats1.append(original_knrm(q_ngram, d_ngram, mu, sigma))

            d_ngram2 = tf.reshape(conv_ne[di], [-1, embedding_dim, filter_numbers])
            feats2.append(original_knrm(q_ngram, d_ngram2, mu, sigma))
    feats1 = tf.concat(feats1, 1)
    feats2 = tf.concat(feats2, 1)

    # Learning-To-Rank layer. o is the final matching score.
    o_pos = (tf.matmul(feats1, W1) + b1)
    o_neg = (tf.matmul(feats2, W1) + b1)

    # regularization+=regularizer2(W1)+regularizer1(W1)+regularizer2(b1)+regularizer1(b1)
    return feats1, feats2, o_pos, o_neg, W1, b1, W_conv1, b_conv1, W_conv2, b_conv2  # ,regularization,W,regularization1,regularization2


def train(optimizer, learning_rate, keep_pro, start=0, batch_num=200, max_prob=0):
    # 增加系数、增加可训练参数
    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())

    f = open('log_train.txt','w')
    for i in range(start, start + batch_num):
        batch_size = 256  ###
        batch = dataset.get_train_batch(batch_size)
        loss_train_batch, acc_train_batch,_ = sess.run([loss, acc,optimizer], feed_dict={
            word_ids_zh: batch[5][0],
            word_ids_pos: batch[5][1],
            word_ids_neg: batch[5][2],
            sequence_lengths_zh: batch[5][3],
            sequence_lengths_pos: batch[5][4],
            sequence_lengths_neg: batch[5][5],
            vob_zh_size: dataset.zh_embedding_matrix,
            vob_en_size: dataset.en_embedding_matrix,
            dropout_pro: keep_pro,
            margin: 1.0,
            initial_learning_rate: learning_rate,
            input_mu: mus,
            input_sigma: sigmas})
        print("Iter " + str(i) + ":train batch loss:" + "{:.6f}".format(
            loss_train_batch) + " train batch acc:" + "{:.6f}".format(acc_train_batch))
        f.write("Iter " + str(i) + ":train batch loss:" + "{:.6f}".format(
                loss_train_batch) + " train batch acc:" + "{:.6f}".format(acc_train_batch) + '\n')

        if i % 500 == 0:
            acc1 = 0.
            acc10 = 0.
            for j in range(len(val_batch_data)):
                score = sess.run(o_pos, feed_dict={
                    word_ids_zh: val_abs_data[j][0],  # 用验证集的数据
                    word_ids_pos: val_abs_data[j][2],
                    sequence_lengths_zh: val_abs_data[j][1],
                    sequence_lengths_pos: val_abs_data[j][3],
                    vob_zh_size: dataset.zh_embedding_matrix,
                    vob_en_size: dataset.en_embedding_matrix,
                    dropout_pro: 1,
                    margin: 1.0,
                    initial_learning_rate: learning_rate,
                    input_mu: mus,
                    input_sigma: sigmas})
                ttl = score.tolist()
                assert len(ttl) == 100  # batch_size256 1个正例 99个负例
                xxl = ttl[0]  # getDoubleData 将正确的放到第一个
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
            print("Iter " + str(i) + ":val top 1 acc:" + "{:.6f}".format(acc1) + " val top 10 acc:" + "{:.6f}".format(
                acc10))
            f.write("Iter " + str(i) + ":val top 1 acc:" + "{:.6f}".format(acc1) + " val top 10 acc:" + "{:.6f}".format(
                acc10) + '\n')
            f.write('\n')
            if acc1 > max_prob:
                print("save the acc" + str(acc1))
                f.write( "save the acc" + str(acc1) + '\n')
                max_prob = acc1
                saver.save(sess, "./conv-knrm6.ckpt")
    f.close()
    return max_prob


def test():
    saver.restore(sess, "./conv-knrm6.ckpt")

    acc1 = 0.
    acc10 = 0.

    f = open('log_test.txt','w')
    for j in range(len(test_batch_data)):
        f.write('item:'+ str(j) + '\n')
        score = sess.run(o_pos,feed_dict={
                word_ids_zh: test_abs_data[j][0],  
                word_ids_pos: test_abs_data[j][2],
                sequence_lengths_zh: test_abs_data[j][1],
                sequence_lengths_pos: test_abs_data[j][3],
                vob_zh_size: dataset.zh_embedding_matrix, 
                vob_en_size: dataset.en_embedding_matrix,
                dropout_pro: 1,
                input_mu: mus,
                input_sigma: sigmas,
                margin: 1})

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
    f.write("test top 1 acc:" + "{:.6f}".format(acc1) + " test top 10 acc:" + "{:.6f}".format(acc10))
    f.close()

'''
optimizer
'''
# training graph
# 分别对正、负例调用model，得分用来计算损失函数
fe_pos, fe_neg, o_pos, o_neg, W1, b1, W_conv1, b_conv1, W_conv2, b_conv2 = conv_knrm_model(W1, b1, W_conv1, b_conv1,
                                                                                           W_conv2, b_conv2,
                                                                                           pretrained_embeddings_1,
                                                                                           pretrained_embeddings_2,
                                                                                           pretrained_embeddings_3, mu,
                                                                                           sigma, filter_numbers)

tv = tf.trainable_variables()
regularization_cost = regularization_rate * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

# margin控制分数差在一定范围内，超过margin则不影响优化
loss = tf.reduce_sum(
    tf.maximum(0.0, margin - o_pos + o_neg)) + regularization_cost  # +regularization#用mean的话相当于慢了100倍 会影响梯度值...
acc = tf.reduce_mean(tf.maximum(tf.sign(o_pos - o_neg), 0.))  ###

# optimizer
global_steps = tf.Variable(0, trainable=False)
decayed_learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                   global_step=global_steps,
                                                   decay_steps=500,
                                                   decay_rate=0.9)
optimizer1 = tf.train.AdamOptimizer(learning_rate=initial_learning_rate, epsilon=1e-05).minimize(loss)
optimizer2 = tf.train.GradientDescentOptimizer(learning_rate=initial_learning_rate).minimize(loss,
                                                                                             global_step=global_steps)
optimizer3 = tf.train.AdamOptimizer(learning_rate=decayed_learning_rate, epsilon=1e-05).minimize(loss)
optimizer4 = tf.train.GradientDescentOptimizer(learning_rate=decayed_learning_rate).minimize(loss,
                                                                                             global_step=global_steps)

'''
run
'''
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.Saver()
sess.run(init)

max_prob = train(optimizer=optimizer2,
                 learning_rate=0.001,
                 keep_pro=0.8,
                 start=0,
                 batch_num=10000)

test()