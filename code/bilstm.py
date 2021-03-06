import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report


class Model:
    """docstring for ClassName"""

    def __init__(self):
        self.EMBED_SIZE = 200
        self.LSTM_HIDDEN_SIZE = 128
        self.VOCAB_SIZE = 76781
        self.ATTENTION_SIZE = 100
        self.HIDDEN_SIZE = 80
        self.DROP = 0.5
        self.TYPE_NUM = 2
        self.LR = 0.001
        self.BATCH_SIZE = 64
        self.should_stop = False
        self.ll = 3
        self.MODEL_PATH = '../model/bilstm/bilstm_model'
        self.MODEL_DIC = '../model/bilstm/'
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.float32, [None, None])
        self.ACC_PATH = '../result/bilstm_acc'
        self.LOSS_PATH = '../result/bilstm_loss'
        self.mode = None
        self.score, self.acc, self.loss, self.train_step = self.run()

    def run(self):
        # 1.Embedding
        # embedd 是词的向量表示
        embed_maxtrix = tf.get_variable('embedding', [self.VOCAB_SIZE, self.EMBED_SIZE],
                                        initializer=tf.random_normal_initializer)
        embedd = tf.nn.embedding_lookup(embed_maxtrix, self.x)


        # 2.bilstm
        ##定义两个lstm
        cell_fw = tf.contrib.rnn.BasicLSTMCell(self.LSTM_HIDDEN_SIZE)
        cell_bw = tf.contrib.rnn.BasicLSTMCell(self.LSTM_HIDDEN_SIZE)
        lstm_out, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=embedd,
                                                      dtype=tf.float32)
        # 取最后一个cell状态作为句子的表示
        lstm_output = tf.concat(lstm_out, 2)
        lstm_output = lstm_output[:, -1, :]

        # 3.Dense Layer
        # lstm_output shape(batch, maxlen, hidd_size)
        with tf.variable_scope('Dense'):
            w = tf.get_variable('weight', [self.LSTM_HIDDEN_SIZE*2, self.TYPE_NUM],
                                initializer=tf.truncated_normal_initializer)
            b = tf.get_variable('bias', [self.TYPE_NUM], initializer=tf.zeros_initializer)
            score = tf.matmul(lstm_output, w) + b

        # 计算准确率
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(score, 1), tf.argmax(self.y, 1)), dtype=tf.float32))

        # 交叉熵计算损失
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=score))

        train_step = tf.train.AdamOptimizer(self.LR).minimize(loss)

        return score, acc, loss, train_step

    def train(self, x_train, y_train, x_dev, y_dev, epoch):
        # 模型的保存和加载
        self.mode = 'train'
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            acc_train_list = []
            acc_test_list = []
            loss_test_list = []
            loss_train_list = []
            step = 0
            es_step = 0
            loss_stop = 99999
            n = 0
            while step < epoch and (self.should_stop is False):
                print('Epoch:{}'.format(step))
                begin = 0
                for i in range(len(x_train) // self.BATCH_SIZE):
                    end = begin + self.BATCH_SIZE
                    x_batch = x_train[begin:end]
                    y_batch = y_train[begin:end]
                    begin = end
                    ###输出训练参数
                    _, acc_train, loss_train, pred = sess.run([self.train_step, self.acc, self.loss, self.score],
                                                              {self.x: x_batch, self.y: y_batch})
                    print('step:{}  [{}/{}]'.format(i, end, len(x_train)))
                    print('acc:{}, loss:{}'.format(acc_train, loss_train))

                acc_t, loss_t = sess.run([self.acc, self.loss], {self.x: x_train[:1000], self.y: y_train[:1000]})
                acc_train_list.append(acc_t)
                loss_train_list.append(loss_t)

                acc_te, loss_te = sess.run([self.acc, self.loss], {self.x: x_dev, self.y: y_dev})
                acc_test_list.append(acc_te)
                loss_test_list.append(loss_te)

                print('Epoch{}----acc:{},loss:{},val-acc:{},val_loss:{}'.format(step, acc_t, loss_t, acc_te, loss_te))
                if loss_te > loss_stop:
                    if n >= self.ll:
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

        # ############绘图###################
        plt.plot(acc_train_list)
        plt.plot(acc_test_list)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(self.ACC_PATH)
        plt.close()

        plt.plot(loss_train_list)
        plt.plot(loss_test_list)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(self.LOSS_PATH)
        plt.close()

    def predict(self, x_test):
        self.mode = 'test'
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(self.MODEL_DIC)  # 找到存储变量值的位置
            saver.restore(sess, ckpt)
            predict = sess.run(self.score, {self.x: x_test})
            return predict

    def verify(self, x_dev, y_dev):
        # 验证集上做一下验证
        dev_predict = self.predict(x_dev)
        dev_predict = np.argmax(dev_predict, axis=1)
        dev_predict = list(dev_predict)
        y_pre = []
        for x in dev_predict:
            y_pre.append(x)
        dev_true = np.argmax(y_dev, axis=1)
        dev_true = list(dev_true)
        y_true = []
        for k in dev_true:
            y_true.append(k)
        target_names = ['False News', 'True News']
        return classification_report(y_true, y_pre, target_names=target_names)

    def output(self, x_test, y_id, path):
        # 结果输出
        test_predict = self.predict(x_test)
        test_predict = np.argmax(test_predict, axis=1)
        test_predict = list(test_predict)
        y_test = []
        for x in test_predict:
            y_test.append(x)
        result = {'id': y_id, 'label': y_test}
        result = pd.DataFrame(result)
        result.to_csv(path, index=False)
        print('output ok!')