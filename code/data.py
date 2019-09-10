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
        self.text_train, self.label, self.text_test, self.text_id = self.load_data()
        self.train_id, self.label, self.test_id, self.word2index, self.seq_len = self.data_process(self.text_train, self.label, self.text_test)
        self.index2word = self.index2word(self.word2index)

    def load_data(self):
        train_data = pd.read_csv(self.train_path, encoding='utf-8')
        test_data = pd.read_csv(self.test_path, encoding='utf-8')
        text_train = train_data['text'].tolist()
        label = train_data['label'].tolist()
        text_test = test_data['text'].tolist()
        text_id = test_data['id'].tolist()
        return text_train, label, text_test, text_id

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

    def get_seqlen(self, train_id):
        seqlen = []
        for x in train_id:
            if len(x) < 300:
                seqlen.append(len(x))
            else:
                seqlen.append(300)
        return seqlen


    def data_process(self, train_data, label, test_data):
        seqlen = []
        x_seg, words = self.segement(train_data)
        test_seg, _ = self.segement(test_data)
        word2index = self.word2index(words)
        train_id = self.word2id(x_seg, word2index)
        seqlen.append(self.get_seqlen(train_id))
        test_id = self.word2id(test_seg, word2index)
        seqlen.append(self.get_seqlen(test_id))
        test_id = pad_sequences(test_id, maxlen=300, padding='post', truncating='post')
        train_id = pad_sequences(train_id, maxlen=300, padding='post', truncating='post')
        label = to_categorical(label, num_classes=2)
        return train_id, label, test_id, word2index, seqlen

    def get_words_id(self, train_id):
        # words_id 包括所有的word
        words_id = []
        for x in train_id:
            words_id.extend(x)
        return words_id

    def get_embedding_set(self, words_id, mode, window_size=None):
        if mode == 'cbow':
            start = 0
            x = []
            y = []
            while start+window_size <= len(words_id):
                x_temp = words_id[start:start + window_size]
                y_temp = [words_id[start + (window_size // 2)]]
                x_temp.pop(window_size // 2)
                x.append(x_temp)
                y.append(y_temp)
                start += 1
            return x, y






