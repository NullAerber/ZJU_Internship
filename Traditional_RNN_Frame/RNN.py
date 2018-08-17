# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# hyper parameter
num_steps = 5
batch_size = 200
num_classes = 2
num_units = 4
learning_rate = 0.1


def gen_data(size=1000000):
    '''
    生成数据
    就是按照文章中提到的规则，这里生成1000000个
    '''
    X = np.array(np.random.choice(2, size=(size,)))
    Y = []
    '''根据规则生成Y'''
    for i in range(size):
        threshold = 0.5
        if X[i - 3] == 1:
            threshold += 0.5
        if X[i - 8] == 1:
            threshold -= 0.25
        if np.random.rand() > threshold:
            Y.append(0)
        else:
            Y.append(1)
    return X, np.array(Y)


def gen_batch(raw_data, batch_size, num_steps):
    # raw_data是使用gen_data()函数生成的数据，分别是X和Y
    raw_x, raw_y = raw_data
    data_length = len(raw_x)

    # 首先将数据切分成batch_size份，0-batch_size，batch_size-2*batch_size。。。
    batch_partition_length = data_length // batch_size
    data_x = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    data_y = np.zeros([batch_size, batch_partition_length], dtype=np.int32)
    for i in range(batch_size):
        data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
        data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]

    # 因为RNN模型一次只处理num_steps个数据，所以将每个batch_size在进行切分成epoch_size份，每份num_steps个数据。注意这里的epoch_size和模型训练过程中的epoch不同。
    epoch_size = batch_partition_length // num_steps

    # x是0-num_steps， batch_partition_length -batch_partition_length +num_steps。。。共batch_size个
    for i in range(epoch_size):
        x = data_x[:, i * num_steps:(i + 1) * num_steps]
        y = data_y[:, i * num_steps:(i + 1) * num_steps]
        yield (x, y)


# 这里的n就是训练过程中用的epoch，即在样本规模上循环的次数
def gen_epochs(n, num_steps):
    for i in range(n):
        yield gen_batch(gen_data(), batch_size, num_steps)


x = tf.placeholder(tf.int32, [batch_size, num_steps], name='input_placeholds')
y = tf.placeholder(tf.int32, [batch_size, num_steps], name='labels_placeholds')

cell = tf.contrib.rnn.BasicRNNCell(num_units=num_units)
# RNN的初始化状态，全设为零。注意state是与input保持一致，接下来会有concat操作，所以这里要有batch的维度。即每个样本都要有隐层状态
h0 = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
# 将输入转化为one-hot编码，两个类别。[batch_size, num_steps, num_classes]
rnn_input = tf.one_hot(x, num_classes)

# 使用dynamic_rnn函数，动态构建RNN模型
outputs, final_state = tf.nn.dynamic_rnn(cell=cell,
                                         inputs=rnn_input,
                                         initial_state=h0,
                                         dtype=tf.float32)

with tf.variable_scope('softmax'):
    W = tf.get_variable('w', [num_units, num_classes])
    b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0, 0))

logits = tf.reshape(
    tf.matmul(tf.reshape(outputs, [-1, num_units]), W) + b,
    [batch_size, num_steps, num_classes])
predictions = tf.nn.softmax(logits)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=predictions)
total_loss = tf.reduce_mean(input_tensor=loss)
train_step = tf.train.AdagradOptimizer(learning_rate).minimize(total_loss)


def train_rnn(num_epochs, num_steps, state_size=4, verbose=True):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros((batch_size, state_size))
            if verbose:
                print("\nEpoch:", idx)

            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_temp, traing_state, _ = sess.run([loss,
                                                                           total_loss,
                                                                           final_state, train_step],
                                                                          feed_dict={x: X, y: Y,
                                                                                     h0: training_state})
                training_loss += training_loss_temp
                if step % 100 == 0 and step > 0:
                    if verbose:
                        print("Average loss at step", step,
                              "for last 100 steps:", training_loss / 100)
                    train_losses.append(training_loss / 100)
                    training_loss = 0

    return train_losses


train_losses = train_rnn(num_epochs=5, num_steps=num_steps, state_size=num_units)
plt.plot(train_losses)
plt.show()
