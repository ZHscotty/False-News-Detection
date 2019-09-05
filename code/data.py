import pandas as pd
import numpy as np
import jieba
import re
import collections
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class Data:
    def __init__(self, train_path=None, test_path=None):
        self.train_path = train_path
        self.test_path = test_path
        self.text_train, self.label, self.text_test = self.load_data()
        self.train_id, self.label, self.test_id, self.word2index = self.data_process(self.text_train, self.label, self.text_test)
        self.index2word = self.index2word(self.word2index)

    def load_data(self):
        train_data = pd.read_csv(self.train_path, encoding='utf-8')
        test_data = pd.read_csv(self.test_path, encoding='utf-8')
        text_train = train_data['text'].tolist()
        label = train_data['label'].tolist()
        text_test = test_data['text'].tolist()
        return text_train, label, text_test

    #分词，并去掉标点符号
    def segement(self, data):
        result = []
        words = []
        for x in data:
            x = re.sub(r'[\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+', '', x)
            seg_result = jieba.lcut(x)
            result.append(seg_result)
            words.extend(seg_result)
        return result, words

    def word2index(self, data):
        word2index = {'PAD': 0, 'UNK': 1}
        words_freqence = collections.Counter(data).most_common()
        for x, _ in words_freqence:
            word2index[x] = len(word2index)
        return word2index

    def word2id(self, data, word2index):
        x_id = []
        for x in data:
            temp = []
            for w in x:
                if w in word2index:
                    temp.append(word2index[w])
                else:
                    temp.append(word2index['UNK'])
            x_id.append(temp)
        return x_id

    def index2word(self, word2index):
        index2word = {}
        for x in word2index:
            index2word[word2index[x]] = x
        return index2word


    def data_process(self, train_data, label, test_data):
        x_seg, words = self.segement(train_data)
        test_seg, _ = self.segement(test_data)
        word2index = self.word2index(words)
        train_id = self.word2id(x_seg, word2index)
        test_id = self.word2id(test_seg, word2index)
        test_id = pad_sequences(test_id, maxlen=300)
        train_id = pad_sequences(train_id, maxlen=300)
        label = to_categorical(label, num_classes=2)
        return train_id, label, test_id, word2index





# if __name__ == '__main__':
#     path = '../data/train.csv'
#     path2 = '../data/test_stage1.csv'
#     data = Data(train_path=path, test_path=path2)
#     index2word = data.index2word
#     test_train = data.text_train[0]
#     print(test_train)
#     data_str = []
#     for x in data.train_id[0]:
#         if index2word[x] != 'PAD':
#             data_str.append(index2word[x])
#     print(data_str)