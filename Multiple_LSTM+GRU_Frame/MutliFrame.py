# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()


# 单层，单向
def signal_layer_one_dim_dynamic_lstm(input_x, n_hidden):
    '''
        返回动态单层LSTM单元的输出，以及cell状态

        args:
            input_x:输入张量  形状为[batch_size,n_steps,n_input]
            n_steps:时序总数
            n_hidden：LSTM单元输出的节点个数 即隐藏层节点数
    '''
    lstm = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden)
    # dynamic_rnn返回两个变量，第一个是每个step的输出值，第二个是最终的状态
    h_outputs, last_state = tf.nn.dynamic_rnn(lstm, input_x, dtype=tf.float32)

    # 注意这里输出需要转置  转换为时序优先的
    # 输出时如果是以[batch_size,max_time,...]形式，即批次优先的矩阵，因为我们需要取最后一个时序的输出，所以需要转置成时间优先的形式。
    h_outputs = tf.transpose(h_outputs, [1, 0, 2])

    return h_outputs, last_state


def single_layer_one_dim_dynamic_gru(input_x, n_hidden):
    '''
    返回动态单层GRU单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''

    gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
    h_outputs, last_state = tf.nn.dynamic_rnn(cell=gru_cell, inputs=input_x, dtype=tf.float32)

    h_outputs = tf.transpose(h_outputs, [1, 0, 2])
    return h_outputs, last_state


# 多层，单向
def multi_layer_dynamic_lstm(input_x, n_hidden):
    '''
    返回动态多层LSTM单元的输出，以及cell状态

    args:
        input_x:输入张量  形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：LSTM单元输出的节点个数 即隐藏层节点数
    '''
    # 可以看做3个隐藏层
    stacked_rnn = []
    for i in range(3):
        stacked_rnn.append(tf.contrib.rnn.LSTMCell(num_units=n_hidden))

    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
    muti_cell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
    hiddens, states = tf.nn.dynamic_rnn(cell=muti_cell, inputs=input_x, dtype=tf.float32)

    # 注意这里输出需要转置  转换为时序优先的
    hiddens = tf.transpose(hiddens, [1, 0, 2])
    return hiddens, states


def multi_layer_dynamic_gru(input_x, n_hidden):
    '''
    返回动态多层GRU单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''
    # 可以看做3个隐藏层
    stacked_rnn = []
    for i in range(3):
        stacked_rnn.append(tf.contrib.rnn.GRUCell(num_units=n_hidden))

    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
    mcell = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
    hiddens, states = tf.nn.dynamic_rnn(cell=mcell, inputs=input_x, dtype=tf.float32)

    # 注意这里输出需要转置  转换为时序优先的
    hiddens = tf.transpose(hiddens, [1, 0, 2])
    return hiddens, states


def multi_layer_dynamic_mix(input_x, n_hidden):
    '''
    返回动态多层GRU和LSTM混合单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''

    # 可以看做2个隐藏层
    gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
    lstm_cell = tf.contrib.rnn.LSTMCell(num_units=n_hidden)

    # 多层RNN的实现 例如cells=[cell1,cell2]，则表示一共有两层，数据经过cell1后还要经过cells
    mcell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell, gru_cell])

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
    hiddens, states = tf.nn.dynamic_rnn(cell=mcell, inputs=input_x, dtype=tf.float32)

    # 注意这里输出需要转置  转换为时序优先的
    hiddens = tf.transpose(hiddens, [1, 0, 2])
    return hiddens, states


# 单层，双向，LSTM
def single_layer_dynamic_bi_lstm(input_x, n_hidden,seq_length):
    '''
    返回单层动态双向LSTM单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''

    # 正向
    lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, forget_bias=1.0)
    # 反向
    lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, forget_bias=1.0)

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出是一个元组 每一个元素也是这种形状
    hiddens, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw_cell, cell_bw=lstm_bw_cell, inputs=input_x,
                                                         sequence_length=seq_length,
                                                     dtype=tf.float32)

    print('hiddens:\n', type(hiddens), len(hiddens), hiddens[0].shape,
          hiddens[1].shape)  # <class 'tuple'> 2 (?, 28, 128) (?, 28, 128)
    # 按axis=2合并 (?,28,128) (?,28,128)按最后一维合并(?,28,256)
    hiddens = tf.concat(hiddens, axis=2)

    # 注意这里输出需要转置  转换为时序优先的
    hiddens = tf.transpose(hiddens, [1, 0, 2])

    return hiddens, state


# 多层，双向，lstm
def multi_layer_dynamic_bi_lstm(input_x, n_hidden):
    '''
    返回多层动态双向LSTM单元的输出，以及cell状态

    args:
        input_x:输入张量 形状为[batch_size,n_steps,n_input]
        n_steps:时序总数
        n_hidden：gru单元输出的节点个数 即隐藏层节点数
    '''
    stacked_fw_rnn = []
    stacked_bw_rnn = []
    for i in range(3):
        # 正向
        stacked_fw_rnn.append(tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, forget_bias=1.0))
        # 反向
        stacked_bw_rnn.append(tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, forget_bias=1.0))
    # muti_fw_lstm = tf.contrib.rnn.MultiRNNCell(stacked_fw_rnn)
    # muti_bw_lstm = tf.contrib.rnn.MultiRNNCell(stacked_bw_rnn)

    # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出为正向和反向合并之后的 即n_hidden*2
    hiddens, fw_state, bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(stacked_fw_rnn, stacked_bw_rnn,
                                                                                 inputs=input_x, dtype=tf.float32)

    print('hiddens:\n', type(hiddens), hiddens.shape)  # <class 'tensorflow.python.framework.ops.Tensor'> (?, 28, 256)

    # 注意这里输出需要转置  转换为时序优先的
    hiddens = tf.transpose(hiddens, [1, 0, 2])

    return hiddens, fw_state, bw_state


def mnist_rnn_classfication(flag):
    '''
    1. load data
    '''
    mnist = input_data.read_data_sets('MNIST-data', one_hot=True)
    print(type(mnist))
    print('Training data shape:', mnist.train.images.shape)  # Training data shape: (55000, 784)
    print('Test data shape:', mnist.test.images.shape)  # Test data shape: (10000, 784)
    print('Validation data shape:', mnist.validation.images.shape)  # Validation data shape: (5000, 784)
    print('Training label shape:', mnist.train.labels.shape)  # Training label shape: (55000, 10)

    '''
    2. define the network structure
    '''
    n_embedding = 28  # LSTM单元输入节点的个数
    n_time_steps = 28  # 序列长度
    n_hidden = 128  # LSTM单元输出节点个数(即隐藏层个数)
    n_classes = 10  # 类别
    batch_size = 128  # 小批量大小
    training_step = 5000  # 迭代次数
    display_step = 200  # 显示步数
    learning_rate = 1e-4  # 学习率
    seq_length = np.zeros([batch_size])
    for i in range(len(seq_length)):
        seq_length[i] = n_time_steps

    input_x = tf.placeholder(tf.float32, [None, n_time_steps, n_embedding])
    input_y = tf.placeholder(tf.float32, [None, n_classes])

    # recurrent the cell
    if flag == 1:
        print('单层动态LSTM网络：')
        h_outputs, fin_state = signal_layer_one_dim_dynamic_lstm(input_x, n_hidden)
    elif flag == 2:
        print('单层动态gru网络：')
        h_outputs, fin_state = single_layer_one_dim_dynamic_gru(input_x, n_hidden)
    elif flag == 3:
        print('多层动态LSTM网络：')
        h_outputs, fin_state = multi_layer_dynamic_lstm(input_x, n_hidden)
    elif flag == 4:
        print('多层动态gru网络：')
        h_outputs, fin_state = multi_layer_dynamic_gru(input_x, n_hidden)
    elif flag == 5:
        print('多层动态LSTM和gru混合网络：')
        h_outputs, fin_state = multi_layer_dynamic_mix(input_x, n_hidden)
    elif flag == 6:
        print('单层动态双向LSTM网络：')
        h_outputs, fin_state = single_layer_dynamic_bi_lstm(input_x, n_hidden,seq_length)
    elif flag == 7:
        print('多层动态双向LSTM网络：')
        h_outputs, fw_state, bw_state = multi_layer_dynamic_bi_lstm(input_x, n_hidden)
    else:
        print('多层动态双向LSTM网络：')
        h_outputs, fw_state, bw_state = multi_layer_dynamic_bi_lstm(input_x, n_hidden)

    # full connection to last output
    # last_state[1] to get the 'h' label array
    # h_outputs[-1] equals to last_state[1]
    outputs = tf.contrib.layers.fully_connected(inputs=h_outputs[-1], num_outputs=n_classes,
                                                activation_fn=tf.nn.softmax)
    '''
    3. set loss fuction
    '''
    # 对数似然损失函数
    # 代价函数 J =-(Σy.logaL)/n    .表示逐元素乘
    cost = tf.reduce_mean(-tf.reduce_sum(input_y * tf.log(outputs), axis=1))

    '''
    4. set optimizer and accuracy value in every 200 step
    '''
    # use adam optimizer to minimize the cost
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # 预测结果评估
    # tf.argmax(output,1)  按行统计最大值得索引
    correct = tf.equal(tf.argmax(outputs, 1), tf.argmax(input_y, 1))  # 返回一个数组 表示统计预测正确或者错误
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))  # 求准确率

    # 创建list 保存每一迭代的结果
    test_accuracy_list = []
    test_cost_list = []

    '''
    5. start train
    '''
    with tf.Session() as sess:
        # 使用会话执行图
        sess.run(tf.global_variables_initializer())  # 初始化变量

        # 开始迭代 使用Adam优化的随机梯度下降法
        for i in range(training_step):
            x_batch, y_batch = mnist.train.next_batch(batch_size=batch_size)
            # Reshape data to get 28 seq of 28 elements
            x_batch = x_batch.reshape([-1, n_time_steps, n_embedding])

            # 开始训练
            optimizer.run(feed_dict={input_x: x_batch, input_y: y_batch})
            if (i + 1) % display_step == 0:
                # 输出训练集准确率
                training_accuracy, training_cost = sess.run([accuracy, cost],
                                                            feed_dict={input_x: x_batch, input_y: y_batch})
                print('Step {0}:Training set accuracy {1},cost {2}.'.format(i + 1, training_accuracy, training_cost))

        # 全部训练完成做测试  分成200次，一次测试50个样本
        # 输出测试机准确率   如果一次性全部做测试，内容不够用会出现OOM错误。所以测试时选取比较小的mini_batch来测试
        for i in range(200):
            x_batch, y_batch = mnist.test.next_batch(batch_size=50)
            # Reshape data to get 28 seq of 28 elements
            x_batch = x_batch.reshape([-1, n_time_steps, n_embedding])

            test_accuracy, test_cost = sess.run([accuracy, cost], feed_dict={input_x: x_batch, input_y: y_batch})
            test_accuracy_list.append(test_accuracy)
            test_cost_list.append(test_cost)
            if (i + 1) % 20 == 0:
                print('Step {0}:Test set accuracy {1},cost {2}.'.format(i + 1, test_accuracy, test_cost))
        print('Test accuracy:', np.mean(test_accuracy_list))


if __name__ == '__main__':
    mnist_rnn_classfication(1)  # 1：单层动态LSTM
    mnist_rnn_classfication(2)  # 2：单层动态gru
    mnist_rnn_classfication(3)  # 3：多层动态LSTM
    mnist_rnn_classfication(4)  # 4：多层动态gru
    mnist_rnn_classfication(5)  # 5: 多层动态LSTM和gru混合网络：
    mnist_rnn_classfication(6)  # 6：单层动态双向LSTM网络：
    mnist_rnn_classfication(7)  # 7：多层动态双向LSTM网络：
