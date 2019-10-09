import data
import bilstm
import bilstm_att_model
import DGCNN_model
import cnn
import cnn_att
import biattention
from sklearn.model_selection import train_test_split
import gensim
import numpy as np
from sklearn.metrics import classification_report
import pickle
import pandas as pd


if __name__ == '__main__':
    path_train = '../data/train.csv'
    path_test = '../data/test_stage1.csv'
    data = data.Data(path_train, path_test)
    seq_len = data.seq_len[0]
    seq_len_test = data.seq_len[1]
    train_id = data.train_id
    label = data.label
    test_id = data.test_id
    word2index = data.word2index
    index2word = data.index2word

    x_train, x_dev, y_train, y_dev = train_test_split(train_id, label, random_state=42, test_size=0.2, stratify=label)
    seqlen_train, seqlen_dev, _, _ = train_test_split(seq_len, label, random_state=42, test_size=0.2, stratify=label)

    # 构建词向量矩阵
    emd = np.random.uniform(-0.05, 0.05, size=(len(word2index), 200))
    model_embedd = gensim.models.word2vec.Word2Vec.load('word2vec_model')
    matrix = model_embedd.wv.load_word2vec_format('../result/w2v')
    for index, x in enumerate(word2index):
        if x in matrix:
            emd[index] = matrix[x]
    print('emd_matrix shape:', emd.shape)
    model = biattention.Model(emd)
    model.train(x_train, y_train, x_dev, y_dev, epoch=20)
    classfication_report = model.verify(x_dev, y_dev)
    print(classfication_report)
    model.output(test_id, data.text_id, '../result/submit.csv')


