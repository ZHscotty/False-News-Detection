import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report


class Model:
    """docstring for ClassName"""

    def __init__(self, index2word=None, embedding_matrix=None):
        self.EMBED_SIZE = 200
        self.VOCAB_SIZE = 76781
        self.ATTENTION_SIZE = 100
        self.HIDDEN_SIZE = 80
        self.FILTER_SIZE = 3
        self.DROP = 0.5
        self.MAXLEN = 300
        self.TYPE_NUM = 2
        self.LR = 0.0001
        self.BATCH_SIZE = 128
        self.FILTER_NUM = 128
        self.FILTER_NUM1 = 150
        self.should_stop = False
        self.ll = 3
        self.MODEL_PATH = '../model/DGCNN/DGCNN_model'
        self.MODEL_DIC = '../model/DGCNN/'
        self.x = tf.placeholder(tf.int32, [None, None])
        self.y = tf.placeholder(tf.float32, [None, None])
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.ACC_PATH = '../result/DGCNN_acc'
        self.LOSS_PATH = '../result/DGCNN_loss'
        self.score, self.acc, self.loss, self.train_step = self.run(index2word, embedding_matrix)

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

    def postion_embedding(self, maxlen, d):
        pos_enc = np.array(
            [[pos / np.power(10000, 2 * j / d) for j in range(d)] if pos != 0 else np.zeros(d) for
             pos in range(maxlen)], dtype=np.float32)
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
        return pos_enc

    def run(self, index2word=None, embedding_matrix=None):
        # 1-1.Embedding
        # embedd 是词的向量表示
        embed_maxtrix = tf.get_variable('embedding', [self.VOCAB_SIZE, self.EMBED_SIZE],
                                        initializer=tf.random_normal_initializer)
        # for i in range(self.VOCAB_SIZE):
        #     if index2word[i] in embedding_matrix:
        #         embedding_matrix[i] = embedding_matrix[index2word[i]]
        embedd = tf.nn.embedding_lookup(embed_maxtrix, self.x)

        # 1-2 CNN
        embedd_expand = tf.expand_dims(embedd, axis=3)
        with tf.variable_scope('CNN_EMB'):
            # 卷积核高， 卷积核长， 输入的通道数， 输出的通道数
            filter_shape = [self.FILTER_SIZE, self.EMBED_SIZE, 1, self.EMBED_SIZE]
            w = tf.get_variable('w', shape=filter_shape, initializer=tf.truncated_normal_initializer)
            b = tf.get_variable('b', shape=[self.EMBED_SIZE], initializer=tf.zeros_initializer)
            conv = tf.nn.conv2d(embedd, w, strides=[1, 1, 1, 1], padding='VALID', dilations=1)
            conv = tf.nn.bias_add(conv, b)
            conv_result = tf.reshape(conv, shape=[-1, self.MAXLEN-self.FILTER_SIZE+1, self.EMBED_SIZE])

        # 1-2-1 attention
        embedd_att = self.attention('emd_att', conv_result, self.EMBED_SIZE, self.EMBED_SIZE)

        # 1-3 postion embedding
        position_embedd = self.postion_embedding(self.MAXLEN, self.EMBED_SIZE)

        embedd_result = tf.add(embedd, embedd_att)
        embedd_result = tf.add(embedd_result, position_embedd)

        # 2 CNN 特征融合
        embedd_result_expand = tf.expand_dims(embedd_result, axis=3)
        with tf.variable_scope('cnn_result'):
            filter_shape1 = [self.FILTER_SIZE, self.EMBED_SIZE, 1, self.FILTER_NUM1]
            w = tf.get_variable('w', shape=filter_shape1, initializer=tf.truncated_normal_initializer)
            b = tf.get_variable('b', shape=[self.FILTER_NUM1])
            conv_mix = tf.nn.conv2d(embedd_result_expand, w, strides=[1, 1, 1, 1], padding='VALID', dilations=1)
            conv_mix = tf.nn.bias_add(conv_mix, b)
            # shape = (batch, maxlen-k+1, attention_size1)
            conv_mix_output = tf.reshape(conv_mix, shape=[-1, self.MAXLEN-self.FILTER_SIZE+1, self.FILTER_NUM1])

        # 3 CNN Layer
        # embedd_result_expand = tf.expand_dims(embedd_result, axis=3)
        dilation_size = [1, 2, 4]
        conv_input = conv_mix_output
        for i in range(len(dilation_size)):
            conv_input_expand = tf.expand_dims(conv_input, axis=3)
            with tf.variable_scope('CNN_{}'.format(i)):
                filter_shape_ = [self.FILTER_SIZE, self.FILTER_NUM1, 1, self.FILTER_NUM]
                w = tf.get_variable('w', shape=filter_shape_, initializer=tf.truncated_normal_initializer)
                b = tf.get_variable('b', shape=[self.FILTER_NUM], initializer=tf.zeros_initializer)
                conv1 = tf.nn.conv2d(conv_input_expand, w, strides=[1, 1, 1, 1], padding='VALID', dilations=dilation_size[i])
                conv1 = tf.nn.bias_add(conv1, b)
                conv_input = tf.reshape(conv1, shape=[-1, conv1.shape[1], self.FILTER_NUM])

        conv_output = conv_input
        # 4 attention layer
        att_output = self.attention('att1', conv_output, self.FILTER_NUM, self.ATTENTION_SIZE)

        # 5 Dense Layer

        with tf.variable_scope('Dense'):
            w = tf.get_variable('weight', [self.ATTENTION_SIZE, self.TYPE_NUM],
                                initializer=tf.truncated_normal_initializer)
            b = tf.get_variable('bias', [self.TYPE_NUM], initializer=tf.zeros_initializer)
            score = tf.matmul(att_output, w) + b

        # 计算准确率
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(score, 1), tf.argmax(self.y, 1)), dtype=tf.float32))

        # 交叉熵计算损失
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=score))

        train_step = tf.train.AdamOptimizer(self.LR).minimize(loss)

        return score, acc, loss, train_step

    def train(self, x_train, y_train, x_dev, y_dev, epoch, seqlen_train=None, seqlen_dev=None):
        # 模型的保存和加载
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
                # 学习率变化
                # if step % 5 == 0 and step > 0:
                # 	LR = LR/2
                begin = 0
                for i in range(len(x_train) // self.BATCH_SIZE):
                    end = begin + self.BATCH_SIZE
                    x_batch = x_train[begin:end]
                    y_batch = y_train[begin:end]
                    #seqlen_batch = seqlen_train[begin:end]
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

    def predict(self, x_test, seqlen_test=None):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.latest_checkpoint(self.MODEL_DIC)  # 找到存储变量值的位置
            saver.restore(sess, ckpt)
            predict = sess.run(self.score, {self.x: x_test})
            return predict

    def verify(self, x_dev, y_dev, seqlen_dev=None):
        # 验证集上做一下验证
        dev_predict = self.predict(x_dev, seqlen_dev)
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