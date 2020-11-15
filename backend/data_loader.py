import os
import re
import sys
import json
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

class NER:

    def __init__(self, data_path, tag_path, pretrained_model_path, use_bert=True):
        self.data_path = data_path
        self.pretrained_model_path = pretrained_model_path
        with open(data_path, 'r') as f:
            self.buf = f.read().strip().split('\n\n')
        with open(data_path, 'r') as f:
            self.texts = f.read()
        with open(tag_path, 'r') as f:
            tmp_buf = f.read()
            self.tag = {x: y for y, x in enumerate(tmp_buf.strip().split('\n'))}
            self.tag_rev = {y: x for y, x in enumerate(tmp_buf.strip().split('\n'))}
        with open('bert/vocab.txt', 'r') as f:
            tmp_buf = f.read()
            self.vocab = {x: y for y, x in enumerate(tmp_buf.strip().split('\n'))}
            self.vocab_rev = {y: x for y, x in enumerate(tmp_buf.strip().split('\n'))}

    def get_train(self, maxlen=None):
        self.train_X = [[self.vocab[line.split()[0]] if line.split()[0] in self.vocab else 100 for line in x.split('\n')][:(99999 if maxlen==None else maxlen)] for x in self.buf]
        self.train_Y = [[self.tag[line.split()[1]] if len(line.split()) > 1 and line.split()[1] in self.tag else 0 for line in x.split('\n')][:(99999 if maxlen==None else maxlen)] for x in self.buf]
        self.maxlen = np.max(np.array([len(x) for x in self.train_X]))
        self.train_X = pad_sequences(self.train_X, self.maxlen, value=0)
        self.train_Y = pad_sequences(self.train_Y, self.maxlen, value=0)
        self.train_Y = to_categorical(self.train_Y)

    def get_test(self, model, maxlen=78):
        self.test_X = [[self.vocab[x] if x in self.vocab else 100 for x in line] for line in self.texts.split('\n')]
        self.test_X = [x[:maxlen] for x in self.test_X if len(x) > 10]
        self.test_Y = model.predict(pad_sequences(self.test_X, maxlen, value=0))
        self.test_sentences = [[self.vocab_rev[x] for x in line] for line in self.test_X]
        self.test_label = [[self.tag_rev[np.argmax(y[1:])+1 if (np.max(y[1:]) > 1e-03) else 0] for y in line[-len(self.test_X[i]):]] for (i, line) in enumerate(self.test_Y)]

def parse_from_raw_txt(path=None, str_=None):

    if not path and not str_:
        raise('you have to specify file path OR input string')
    if path and str_:
        raise('you can only specify file path OR input string')

    words = []

    if path:

        with open(path, 'r') as fp:
            for buf in fp:
                re_all = re.compile(u'[\u4e00-\u9fa5|\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5|\n]+')
                #re_words = re.compile(u'[\u4e00-\u9fa5]+')
                res = re.findall(re_all, buf)
                for i in res:
                    if len(i) > 1:
                        for j in i:
                            if j != '\n' and j != '。' and j != '”' and j != '“':
                                words.append(j)
                            elif j != '\n':
                                words.append('。')
                    else:
                        if i != '\n':
                            words.append(i)
                        elif len(words) and words[-1] != '。':
                            words.append('。')
    else:
        for buf in str_:
            re_all = re.compile(u'[\u4e00-\u9fa5|\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5|\n]+')
            res = re.findall(re_all, buf)
            for i in res:
                for j in i:
                    if j != '\n' and j != '”' and j != '“':
                        words.append(j)

    words = ''.join(words)

    res = ''
    for word in words:
        res += word
        if word == '。':
            res += '\n'
    if res[-1] != '。':
        res += '。'

    return res