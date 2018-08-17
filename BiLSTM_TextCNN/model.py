#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os


class TextCNN():
    def __init__(self, dataset, filter_numbers, sentence_length,lstm_n_hidden, lstm_n_layer, cnn_layer_size_list,
                 learn_rate, epochs, batch_size, dropout_pro):
        tf.reset_default_graph()

        '''
        0. init variable
        '''
        self.name = str(filter_numbers) + '_' + str(lstm_n_hidden) + '+' + str(cnn_layer_size_list) + '+' + str(
            learn_rate) + '_' + str(batch_size) + '_' + str(dropout_pro) + '+++'
        self.name = self.name.replace('[', '').replace(']', '')
        self.learning_rate = learn_rate
        self.epochs = epochs
        self.seq_length = sentence_length
        self.filter_numbers = filter_numbers
        self.batch_size = batch_size
        self.dropout_pro_item = dropout_pro
        self.embedding_dim = len(dataset.zh_embedding_matrix[0])
        self.zh_vocab_size = len(dataset.zh_embedding_matrix)
        self.en_vocab_size = len(dataset.en_embedding_matrix)
        self.lstm_n_hidden = lstm_n_hidden
        self.lstm_n_layer = lstm_n_layer
        self.cnn_layer_size = cnn_layer_size_list

        self.dataset = dataset
        self.keep_prob = tf.placeholder(tf.float32)

        '''
        1. input layer
        '''
        #
        self.input_zh = tf.placeholder(tf.int32, shape=[None, sentence_length])
        self.input_pos = tf.placeholder(tf.int32, shape=[None, sentence_length])
        self.input_neg = tf.placeholder(tf.int32, shape=[None, sentence_length])

        # self.input_en_image = tf.placeholder()

        self.zh_seq_len = tf.placeholder(tf.int32, shape=[None])
        self.pos_seq_len = tf.placeholder(tf.int32, shape=[None])
        self.neg_seq_len = tf.placeholder(tf.int32, shape=[None])

        self.zh_dict_vec = tf.placeholder(shape=[self.zh_vocab_size, self.embedding_dim],
                                          dtype=tf.float32)
        self.en_dict_vec = tf.placeholder(shape=[self.en_vocab_size, self.embedding_dim],
                                          dtype=tf.float32)

        '''
        2. embedding layer
        '''
        with tf.name_scope('embedding_layer'):
            embeddings_zh = tf.nn.embedding_lookup(self.zh_dict_vec, self.input_zh)
            embeddings_pos = tf.nn.embedding_lookup(self.en_dict_vec, self.input_pos)
            embeddings_neg = tf.nn.embedding_lookup(self.en_dict_vec, self.input_neg)

        '''
        3. use BiLSTM to zh and pos/neg 
        hiddens = [batch_size, seq_length, n_hiddens]
        '''
        # zh forward and backward lstm
        zh_lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_n_hidden, forget_bias=1.0)
        zh_lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_n_hidden, forget_bias=1.0)

        # en(pos/neg) backward layer
        en_lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_n_hidden, forget_bias=1.0)
        en_lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_n_hidden, forget_bias=1.0)

        with tf.variable_scope("zh_bilstm", reuse=None) as scope:
            hiddens_zh, state_zh = tf.nn.bidirectional_dynamic_rnn(cell_fw=zh_lstm_fw_cell,
                                                                   cell_bw=zh_lstm_bw_cell,
                                                                   inputs=embeddings_zh,
                                                                   sequence_length=self.zh_seq_len,
                                                                   dtype=tf.float32)
            hiddens_zh = tf.concat(hiddens_zh, axis=2)

        with tf.variable_scope("en_bilstm", reuse=None) as scope:
            hiddens_pos, state_pos = tf.nn.bidirectional_dynamic_rnn(cell_fw=en_lstm_fw_cell,
                                                                     cell_bw=en_lstm_bw_cell,
                                                                     inputs=embeddings_pos,
                                                                     sequence_length=self.pos_seq_len,
                                                                     dtype=tf.float32)
            hiddens_pos = tf.concat(hiddens_pos, axis=2)

        with tf.variable_scope("en_bilstm", reuse=True) as scope:
            hiddens_neg, state_neg = tf.nn.bidirectional_dynamic_rnn(cell_fw=en_lstm_fw_cell,
                                                                     cell_bw=en_lstm_bw_cell,
                                                                     inputs=embeddings_neg,
                                                                     sequence_length=self.neg_seq_len,
                                                                     dtype=tf.float32)

            hiddens_neg = tf.concat(hiddens_neg, axis=2)

        '''
        4. concat zh outputs and pos/neg outputs
        '''
        self.pos_concat = tf.concat([hiddens_zh, hiddens_pos], -1)
        self.neg_concat = tf.concat([hiddens_zh, hiddens_neg], -1)
        self.pos_concat = tf.expand_dims(self.pos_concat, -1)
        self.neg_concat = tf.expand_dims(self.neg_concat, -1)

        '''
        5. conv layer + maxpool layer for each filer size
        '''
        pool_layer_list_pos = []
        pool_layer_list_neg = []
        W_list = []
        b_list = []
        # convolutio layer
        for layer in cnn_layer_size_list:
            filter_shape = [layer, self.lstm_n_hidden * 4, 1, filter_numbers]
            W_list.append(tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='filter_weight'))
            b_list.append(tf.Variable(tf.constant(0.1, shape=[filter_numbers]), name='filter_bias'))

        for W, b, filter_size in zip(W_list, b_list, cnn_layer_size_list):
            max_pool_layer_pos = self.__add_conv_layer(self.pos_concat, W, b, filter_size)
            pool_layer_list_pos.append(max_pool_layer_pos)
            max_pool_layer_neg = self.__add_conv_layer(self.neg_concat, W, b, filter_size)
            pool_layer_list_neg.append(max_pool_layer_neg)

        '''
        6. full connect droput + softmax + l2
        '''
        # combine all the max pool —— feature
        with tf.name_scope('dropout_layer'):
            max_num = len(cnn_layer_size_list) * self.filter_numbers

            h_pool_pos = tf.concat(pool_layer_list_pos, name='last_pool_layer', axis=3)
            pool_layer_flat_pos = tf.reshape(h_pool_pos, [-1, max_num], name='pool_layer_flat')
            dropout_pro_layer_pos = tf.nn.dropout(pool_layer_flat_pos, self.keep_prob, name='dropout')

            h_pool_neg = tf.concat(pool_layer_list_neg, name='last_pool_layer', axis=3)
            pool_layer_flat_neg = tf.reshape(h_pool_neg, [-1, max_num], name='pool_layer_flat')
            dropout_pro_layer_neg = tf.nn.dropout(pool_layer_flat_neg, self.keep_prob, name='dropout')

        with tf.name_scope('full_con_layer'):
            self.l2_loss = 0
            num_out = 1
            SoftMax_W = tf.Variable(tf.truncated_normal([max_num, num_out], stddev=0.01), name='softmax_linear_weight')
            SoftMax_b = tf.Variable(tf.constant(0.1, shape=[num_out]), name='softmax_linear_bias')
            self.l2_loss += tf.nn.l2_loss(SoftMax_W)
            self.l2_loss += tf.nn.l2_loss(SoftMax_b)
            self.softmax_values_pos = tf.nn.xw_plus_b(dropout_pro_layer_pos, SoftMax_W, SoftMax_b,
                                                      name='soft_values')
            self.softmax_values_neg = tf.nn.xw_plus_b(dropout_pro_layer_neg, SoftMax_W, SoftMax_b,
                                                      name='soft_values')

            self.score_pos = self.softmax_values_pos
            self.score_neg = self.softmax_values_neg

        '''
        7. calculate loss
        '''
        with tf.name_scope('loss'):
            self.margin = tf.placeholder(tf.float32)
            self.loss = tf.reduce_sum(tf.maximum(0.0, self.margin - self.score_pos + self.score_neg))

        '''
        8. calcalate accuarcy
        '''
        with tf.name_scope('accuracy'):
            self.acc = tf.reduce_mean(tf.maximum(tf.sign(self.score_pos - self.score_neg), 0.))

        '''
        9. use optimizer
        '''
        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session()
        self.saver = tf.train.Saver()

    def __add_conv_layer(self, concat, W, b, filter_size):
        with tf.name_scope('conv_maxpool_layer'):
            # 参数说明
            # 第一个参数input：指需要做卷积的输入图像 [训练时一个batch的图片数量, 图片高度, 图片宽度, 图像通道数]
            # 第二个参数filter：相当于CNN中的卷积核 [卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]
            # 第三个参数strides：卷积时在图像每一维的步长，这是一个一维的向量，长度4,
            # 第四个参数padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式
            # 第五个参数：use_cudnn_on_gpu: bool类型，是否使用cudnn加速，默认为true
            conv_layer = tf.nn.conv2d(concat, W, strides=[1, 1, 1, 1], padding='VALID',
                                      name='conv_layer')
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, b), name='relu_layer')

            max_pool_layer = tf.nn.max_pool(relu_layer, ksize=[1, self.seq_length - filter_size + 1, 1, 1],
                                            strides=[1, 1, 1, 1], padding='VALID', name='maxpool')
            return max_pool_layer

    def train(self):
        os.mkdir('./model/' + self.name)
        f = open('./log/' + self.name + 'log_train.txt', 'w')
        self.session.run(tf.global_variables_initializer())
        # self.session = tf_debug.LocalCLIDebugWrapperSession(sess=self.session)
        # 迭代训练
        min_loss = 9999999
        max_acc = 0.0
        for epoch in range(self.epochs):
            train_batch = self.dataset.get_train_batch(self.batch_size)
            feed = {
                self.input_zh: train_batch[5][0],
                self.input_pos: train_batch[5][1],
                self.input_neg: train_batch[5][2],
                self.zh_seq_len: train_batch[5][3],
                self.pos_seq_len: train_batch[5][4],
                self.neg_seq_len: train_batch[5][5],
                self.zh_dict_vec: self.dataset.zh_embedding_matrix,
                self.en_dict_vec: self.dataset.en_embedding_matrix,
                self.margin: 1.0,
                self.keep_prob: self.dropout_pro_item
            }
            _, loss, accuracy, s_pos, s_neg = self.session.run(
                [self.optimizer, self.loss, self.acc, self.score_pos, self.score_neg], feed_dict=feed)
            f.write('Epoch {:>3} train_loss = {} accuracy = {} + \n'.
                    format(epoch, loss, accuracy))
            if epoch % 50 == 0:
                print('Epoch {:>3} train_loss = {} accuracy = {} + \n'.
                      format(epoch, loss, accuracy))
            if accuracy > max_acc:
                print('updata new model')
                if accuracy != 1.0:
                    max_acc = accuracy
                    min_loss = loss
                self.saver.save(self.session,
                                './model/' + self.name + '/' + str(epoch) + '_' + str(loss) + '_' + str(
                                    accuracy) + '.ckpt')
        self.saver.save(self.session,
                        './model/' + self.name + '/last epoch' + '_' + str(loss) + '_' + str(
                            accuracy) + '.ckpt')

    def val(self, epoch):
        val_batch_data, val_abs_data = self.dataset.get_val_data(batch_size=129)
        acc_top1 = 0.
        acc_top10 = 0.

        for j in range(len(val_batch_data)):
            score = self.session.run(self.score_pos, feed_dict={
                self.input_zh: val_abs_data[j][0],
                self.input_pos: val_abs_data[j][2],
                self.zh_seq_len: val_abs_data[j][1],
                self.pos_seq_len: val_abs_data[j][3],
                self.zh_dict_vec: self.dataset.zh_embedding_matrix,
                self.en_dict_vec: self.dataset.en_embedding_matrix,
                self.keep_prob: 1.0,
                self.margin: 1.0})

            assert len(score) == 100
            n = 0
            for k in range(1, 100):
                if score[0] <= score[k]:  # score越大越好
                    n += 1
            if n == 0:
                acc_top1 += 1
            if n < 10:
                acc_top10 += 1
        acc_top1 = float(acc_top1) / float(len(val_batch_data))
        acc_top10 = float(acc_top10) / float(len(val_batch_data))
        print("epoch = {} val top 1 acc = {}  test top 10 acc:={}".format(epoch, acc_top1, acc_top10))

        return acc_top1

    def test(self, model_name):
        test_batch_data, test_abs_data = self.dataset.get_test_data()
        acc_top1 = 0.
        acc_top10 = 0.

        self.saver.restore(self.session, 'model/' + self.name + '/' + model_name + '.cpkt')

        f = open('./log/' + self.name + model_name + 'log_test.txt', 'w')

        for j in range(len(test_batch_data)):
            score = self.session.run(self.score_pos, feed_dict={
                self.input_zh: test_abs_data[j][0],
                self.input_pos: test_abs_data[j][2],
                self.input_zh_title: test_abs_data[j][4],
                self.input_pos_title: test_abs_data[j][5],
                self.zh_seq_len: test_abs_data[j][1],
                self.pos_seq_len: test_abs_data[j][3],
                self.zh_dict_vec: self.dataset.zh_embedding_matrix,
                self.en_dict_vec: self.dataset.en_embedding_matrix,
                self.keep_prob: 1.0,
                self.margin: 1.0})
            assert len(score) == 100

            f.write('batch :' + str(j) + '\n')
            print('batch :' + str(j))
            f.write('score[0]: ' + str(score[0]) + '\n')
            print('score[0]: ' + str(score[0]))

            n = 0
            for k in range(1, 100):
                if score[0] <= score[k]:  # score越大越好
                    n += 1
                    print(str(score[k]))

                f.write(str(score[k]) + '\n')

            if n == 0:
                acc_top1 += 1
                print(acc_top1)
            if n < 10:
                acc_top10 += 1
                print(acc_top10)

            f.write('\n')
            print('')

        f.write('acc_top1=' + str(acc_top1) + '    ' + 'acc_top10=' + str(
            acc_top10) + '       ' + 'len(test_batch_data)' + str(len(test_batch_data)))
        print('acc_top1=' + str(acc_top1) + '    ' + 'acc_top10=' + str(
            acc_top10) + '       ' + 'len(test_batch_data)' + str(len(test_batch_data)))

        acc_top1 = float(acc_top1) / float(len(test_batch_data))
        acc_top10 = float(acc_top10) / float(len(test_batch_data))
        f.write("test top 1 acc = {}  test top 10 acc:={}".format(acc_top1, acc_top10))
        f.close()
        print("test top 1 acc = {}  test top 10 acc:={}".format(acc_top1, acc_top10))

    def close(self):
        self.session.close()



if __name__ == '__main__':
    # step 1: load data
    print("Loading training and validation data...")
    dataset = DataSet.DataSet()

    print('Configuring CNN model...')
    # step3 create TextCNN model
    text_cnn = TextCNN(dataset=dataset,
                        filter_numbers=128,
                        sentence_length=100,
                        lstm_n_hidden=64,
                        lstm_n_layer=3,
                        cnn_layer_size_list=[3, 4, 5],
                        learn_rate=1e-3,
                        epochs=5000,
                        batch_size=256,
                        dropout_pro=0.6)
    # step4 start train
    text_cnn.train()
    # step5 test
    text_cnn.test()

    text_cnn.close()