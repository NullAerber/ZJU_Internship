# ZJU Internship
Some done work in ZJU during the internship period. Most about neural network frame using tensorflow.

## Environment
Python: 2.7 (partly support python 3.6 )

Tensorflow: 1.0.1

## 1. Traditional RNN Frame
### Input :
A generated sequence under some rules.
### Output :
To predict the next output.
### Model :
Traditional RNN model.

#### reference:
[深度学习（07）_RNN-循环神经网络-02-Tensorflow中的实现](https://blog.csdn.net/u013082989/article/details/73469095/)



## 2. Multiple LSTM/GRU Frame
### Input :
The image of mnist.
### Output :
The classification of number from 0 to 10.
### Model :
Include traditional LSTM、traditional GRU、multiple Layer LSTM、multiple Layer GRU、traditional Bi-Directional LSTM、Multiple Layers BiLSTM model.

#### PS.:
The implement of different models using tensorflow framework.

To be familiar with the specific details of the implementation.
#### reference:
1. [使用TensorFlow实现LSTM和GRU网络](https://www.cnblogs.com/zyly/p/9029591.html)
2. [零基础入门深度学习(6) - 长短时记忆网络(LSTM)](https://www.zybuluo.com/hanbingtao/note/581764)
3. [深入理解LSTM记忆元网络](https://jellycsc.github.io/2018/04/13/understanding-lstm-networks/)

## 3. TextCNN
### Input :
The sequence of a movie review.
### Output :
The emotion tendencies, positive or negative.
### Model :
embedding layer =>Multiple convoluted layer with max-pooling layer =>Desor layer =>Desor layer =>softmax layer

#### PS.:

1. There are two version TextCNN, you can run 
```
python run_textcnn_model_v1.py
```
or 
```
python run_textcnn_model_v2.py
```
to get different version of textcnn.

the difference is the detail implementation in these two model.

2. the training data download url for word embedding is here : http://mattmahoney.net/dc/text8.zip


## 4. BiLSTM-TextCNN
### Aim:
To match the entities between different baidu baike and wikipedia.
### Input:
Some element of one baidu baiku entity and 100 wikipedia entity candidates.
### Output:
The wikipedia entity which has highest score.
### Model:
1. Use triple training.

2. embedding layer => bilstm layer => concat layer => textcnn layer => densor layer => score

![model1.png](model1.png)
### Result
hyper parameter| Train accuracy | Val accuracy | Test accuracy 
----|---------|--------|------
Filter number =16，Bath size =256 |96.8% |Top1：21.7% Top10：65.8% |Top1：13.5% Top10：60%
Filter number =128，Bath size =256 |100% |Top1：21.7% Top10：71.3% |Top1：17% Top10：64.4%
Filter number =64，Bath size =128 |98.4% |Top1：14.7% Top10：65.1% |Top1：14.4% Top10：60.2%
Filter number =64，Bath size =32 |100% |Top1：13.9% Top10：46.5% |Top1：9.25% Top10：48.2%

#### PS. :
The data is provided by Zhejiang University DCD lab. And this data can not be public, so i just push the model code.


## 5. Attention-BiLSTM-TextCNN
Under the base of **BiLSTM-TextCNN**, add one attention base model layer before bilstm.But the result is not good. Compared with the previous model, the accuracy rate dropped by 10%.


## 6. Attention mechanism for text classification tasks

Tensorflow implementation of attention mechanism for text classification tasks.  
Inspired by "Hierarchical Attention Networks for Document Classification", Zichao Yang et al. (http://www.aclweb.org/anthology/N16-1174).

### Note:
This is fork from other's.

https://github.com/ilivans/tf-rnn-attention

I edit some code to make this project can run on python 3.6.

### Extend
1. [深度解析注意力模型(attention model) --- image_caption的应用](https://segmentfault.com/a/1190000011744246)
2. [heuritech.com - ATTENTION MECHANISM](https://blog.heuritech.com/2016/01/20/attention-mechanism/)
3. [浅谈Attention-based Model【原理篇】](https://blog.csdn.net/wuzqchom/article/details/75792501)

## 7. Conv-K-NRM
This is the Tensorflow implementation of **Convolutional Neural Networks for Soft-Matching N-Grams in Ad-hoc Search** which completed by my friend **陈璐@Zhongnan University**.

## 8. BiMPM + CNN(check unfinished )
This is the Tensorflow implementation of **Bilateral Multi-Perspective Matching for Natural Language Sentences**which completed by my friend **郭悦@Zhongshan University**. And based on this paper， my friend add one CNN layer to increase the accuracy about 8%. 