import data
import cbow
from sklearn.model_selection import train_test_split
import logging
import gensim


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # path_train = '../data/train.csv'
    # path_test = '../data/test_stage1.csv'
    # data = data.Data(path_train, path_test)
    # word2index = data.word2index
    # # index2word = data.index2word
    # text_train = data.text_train
    # x_seg, _ = data.segement(text_train)
    # train_id = data.word2id(x_seg, word2index)
    #words_id = data.get_words_id(train_id)
    #x, y = data.get_embedding_set(words_id, 'cbow', window_size=3)
    # x_train, x_dev, y_train, y_dev = train_test_split(x, y, random_state=42, test_size=0.2)
    # model = cbow.Model()
    # model.train(x_train, y_train, x_dev, y_dev, epoch=10)
    # embedding_matrix = model.predict()
    # model.similarity(embedding_matrix, index2word)
    model = gensim.models.word2vec.Word2Vec(sentences=x_seg, min_count=3, size=200)
    #model.save('word2vec_model')
    model = gensim.models.word2vec.Word2Vec.load('word2vec_wx')
    #model.wv.save_word2vec_format('../result/w2v', binary=False)
    matrix = model.wv.load_word2vec_format('../result/w2v')