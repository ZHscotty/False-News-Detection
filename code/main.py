import data
import bilstm_att_model
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import classification_report
import pickle
import pandas as pd


if __name__ == '__main__':
    path_train = '../data/train.csv'
    path_test = '../data/test_stage1.csv'
    data = data.Data(path_train, path_test)
    train_id = data.train_id
    label = data.label
    test_id = data.test_id
    word2index = data.word2index
    index2word = data.index2word

    x_train, x_dev, y_train, y_dev = train_test_split(train_id, label, random_state=42, test_size=0.2, stratify=label)

    model = bilstm_att_model.Model()
    #model.train(x_train, y_train, x_dev, y_dev, epoch=20)
    classfication_report = model.verify(x_dev, y_dev)
    print(classfication_report)
    model.output(test_id, data.text_id, '../result/submit.csv')
