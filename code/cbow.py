import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math
import random


class Model(object):
    """docstring for ClassName"""

    def __init__(self):
        self.EMBED_SIZE = 200
        self.LSTM_HIDDEN_SIZE = 128
        self.VOCAB_SIZE = 76781
        self.HIDDEN_SIZE = 100
        self.DROP_DENSE = 0.5
        self.LR = 0.001
        self.BATCH_SIZE = 128
        self.should_stop = False
        self.ll = 3
        self.MODEL_PATH = '../model/cbow/cbow_model'
        self.MODEL_DIC = '../model/cbow/'
        self.ACC_PATH = '../result/cbow_acc.png'
        self.LOSS_PATH = '../result/cbow_loss.png'
        self.emb_drop = False
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.int64, [None, None])
        self.loss, self.train_step, self.embedding_matrix = self.run()

    def run(self):
        # 1.Dense1_layer
        # input x=(batch, window_size, vocab_size)
        with tf.variable_scope('Dense1'):
            w = tf.get_variable('embedding_martix', [self.VOCAB_SIZE, self.EMBED_SIZE],
                                initializer=tf.random_normal_initializer(-1, 1))
            embedd = tf.nn.embedding_lookup(w, self.x)
            # b = tf.get_variable('bias', [self.HIDDEN_SIZE], initializer=tf.zeros_initializer())
            embedding_matrix = w
            # print('embedd shape', embedd.shape)
            nce_weights = tf.Variable(
                tf.truncated_normal([self.VOCAB_SIZE, self.EMBED_SIZE],
                                    stddev=1.0 / math.sqrt(self.EMBED_SIZE)))
            nce_biases = tf.Variable(tf.zeros([self.VOCAB_SIZE]))

        # 2.上下文表示concat
        target = tf.reduce_sum(embedd, axis=1)


        # 准确率
        #acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(score, 2), self.y), tf.float32))
        # 交叉熵计算损失
        # loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=score)
        # loss = tf.reduce_mean(loss)
        loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases, inputs=target, labels=self.y, num_sampled=64,
                           num_classes=self.VOCAB_SIZE))

        train_step = tf.train.AdamOptimizer(self.LR).minimize(loss)

        return loss, train_step, embedding_matrix

    def train(self, x_train, y_train, x_dev, y_dev, epoch):
        # 模型的保存和加载
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            loss_test_list = []
            loss_train_list = []
            # 表示训练的轮数
            step = 0
            # 表示earlystop的轮数
            es_step = 0
            # 最小损失
            loss_stop = 99999
            # 表示loss连续变大的轮数
            n = 0
            while step < epoch and (self.should_stop is False):
                print('Epoch:{}'.format(step))
                # 学习率变化
                # if step % 5 == 0 and step > 0:
                # 	LR = LR/2
                begin = 0
                for i in range(len(x_train) // self.BATCH_SIZE):
                    end = begin + self.BATCH_SIZE
                    x_batch = x_train[begin:end]
                    y_batch = y_train[begin:end]
                    begin = end
                    ###输出训练参数
                    # print('x_batch shape:', x_batch.shape)
                    # print('y_batch shape', y_batch.shape)
                    _, loss_train = sess.run([self.train_step, self.loss], {self.x: x_batch, self.y: y_batch})
                    print('step:{}  [{}/{}]'.format(i, end, len(x_train)))
                    print('loss:{}'.format(loss_train))

                loss_t = sess.run(self.loss, {self.x: x_train[:10000], self.y: y_train[:10000]})
                loss_train_list.append(loss_t)

                loss_te = sess.run(self.loss, {self.x: x_dev[:10000], self.y: y_dev[:10000]})
                loss_test_list.append(loss_te)

                print('Epoch{}----loss:{},val_loss:{}'.format(step, loss_t, loss_te))
                if loss_te > loss_stop:
                    if n > self.ll:
                        self.should_stop = True
                    else:
                        n += 1
                else:
                    saver.save(sess, self.MODEL_PATH)
                    es_step = step
                    n = 0
                    loss_stop = loss_te
                step += 1

            if self.should_stop:
                print('Early Stop at Epoch{}'.format(es_step))

        #############绘图###################
        plt.plot(loss_train_list)
        plt.plot(loss_test_list)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(self.LOSS_PATH)
        plt.close()

    def predict(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(self.MODEL_DIC)  # 找到存储变量值的位置
            saver.restore(sess, ckpt)
            embedingmatrix = sess.run(self.embedding_matrix)
            return embedingmatrix

    def similarity(self, embeddingmatrix, index2word):
        word_num = 50
        radom_index = []
        for i in range(word_num):
            radom_index.append(random.randint(0, self.VOCAB_SIZE))
        embeddingmatrix = embeddingmatrix[radom_index]
        similarity = np.matmul(embeddingmatrix, np.transpose(embeddingmatrix))
        for i in range(word_num):
            word = index2word[radom_index[i]]
            near = (-similarity[i]).argsort()[1:10]
            # print(near)
            near_word = []
            for x in near:
                near_word.append(index2word[x])
            print('word:', word, 'nearest words:', near_word)
