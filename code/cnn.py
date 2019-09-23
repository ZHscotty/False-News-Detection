import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report


class Model:
    """docstring for ClassName"""

    def __init__(self):
        self.EMBED_SIZE = 200
        self.VOCAB_SIZE = 76781
        self.HIDDEN_SIZE = 80
        self.DROP = 0.5
        self.TYPE_NUM = 2
        self.FILTER_NUM = 150
        self.LR = 0.0003
        self.FILTER_SIZE = 3
        self.MAXLEN = 300
        self.BATCH_SIZE = 64
        self.should_stop = False
        self.ll = 3
        self.MODEL_PATH = '../model/cnn/cnn_model'
        self.MODEL_DIC = '../model/cnn/'
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.float32, [None, None])
        self.ACC_PATH = '../result/cnn_acc'
        self.LOSS_PATH = '../result/cnn_loss'
        self.mode = None
        self.score, self.acc, self.loss, self.train_step = self.run()

    def attention(self, scope_name, input_tensor, input_size, output_size):
        with tf.variable_scope(scope_name):
            # Q
            wq = tf.get_variable('weightq', [input_size, output_size],
                                 initializer=tf.truncated_normal_initializer)
            bq = tf.get_variable('biasq', [output_size], initializer=tf.zeros_initializer)
            q = tf.einsum('aij,jk->aik', input_tensor, wq) + bq

            # K
            wk = tf.get_variable('weightk', [input_size, output_size],
                                 initializer=tf.truncated_normal_initializer)
            bk = tf.get_variable('biask', [output_size], initializer=tf.zeros_initializer)
            k = tf.einsum('aij,jk->aik', input_tensor, wk) + bk

            # V
            wv = tf.get_variable('weightv', [input_size, output_size],
                                 initializer=tf.truncated_normal_initializer)
            bv = tf.get_variable('biasv', [output_size], initializer=tf.zeros_initializer)
            v = tf.einsum('aij,jk->aik', input_tensor, wv) + bv

            # Q*K/
            k = tf.transpose(k, perm=[0, 2, 1])
            # shape(batch, maxlen, maxlen)
            input_score = tf.einsum('aij,ajk->aik', q, k)
            d = tf.sqrt(float(output_size))
            input_score = input_score / d
            input_score = tf.nn.softmax(input_score, axis=2)

            # get weighted value
            # shape(batch, maxlen, 2*lstm_hidden_size)
            att_output = tf.einsum('aij,ajk->aik', input_score, v)

            # 加起来作为句子的表示
            att_output = tf.reduce_mean(att_output, axis=1)
            return att_output

    def run(self):
        # 1.Embedding
        # embedd 是词的向量表示
        embed_maxtrix = tf.get_variable('embedding', [self.VOCAB_SIZE, self.EMBED_SIZE],
                                        initializer=tf.random_normal_initializer)
        # shape = (batch, maxlen, embedding_size)
        embedd = tf.nn.embedding_lookup(embed_maxtrix, self.x)
        embedd = tf.expand_dims(embedd, axis=3)


        # 2.cnn
        filter_shape = [self.FILTER_SIZE, self.EMBED_SIZE, 1, self.FILTER_NUM]
        with tf.variable_scope('cnn'):
            w = tf.get_variable('weight', shape=filter_shape, initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('bias', shape=[self.FILTER_NUM], initializer=tf.zeros_initializer())
            cnn = tf.nn.conv2d(embedd, w, strides=[1, 1, 1, 1], padding='VALID')
            # shape = (batch, maxlen-k+1, 1, filter_num)
            cnn = tf.nn.bias_add(cnn, b)
        # shape = (batch, maxlen-k+1, filter_num)
        cnn_output = tf.reshape(cnn, shape=[-1, self.MAXLEN-self.FILTER_SIZE+1, self.FILTER_NUM])
        cnn_output = tf.reduce_mean(cnn_output, axis=1)


        # 3.Dense Layer
        # lstm_output shape(batch, maxlen, hidd_size)
        with tf.variable_scope('Dense'):
            w = tf.get_variable('weight', [self.FILTER_NUM, self.TYPE_NUM],
                                initializer=tf.truncated_normal_initializer)
            b = tf.get_variable('bias', [self.TYPE_NUM], initializer=tf.zeros_initializer)
            score = tf.matmul(cnn_output, w) + b

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
                    es_loss = loss_te
                    es_acc = acc_te
                    es_step = step
                    n = 0
                    loss_stop = loss_te

                step += 1

            if self.should_stop:
                print('Early Stop at Epoch{} acc:{} loss:{}'.format(es_step, es_loss, es_acc))

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