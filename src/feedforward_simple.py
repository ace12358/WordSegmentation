#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Usage:
python feed_forward_simple.txt config.ini
'''
import sys
import chainer
import numpy as np
import configparser
import chainer.functions as F
from chainer import optimizers


def show_config(ini):
    '''
    show config
    '''
    for section in ini.sections():
        print ('[%s]' % (section))
        show_sectoin(ini, section)
    return


def create_vocab():
    vocab = dict()
    for f in [train_file, test_file]:
        for line in open(f):
            for char in ''.join(line.strip().split()):
                if char not in vocab:
                    vocab[char] = len(vocab)
    vocab['<s>'] = len(vocab)
    vocab['</s>'] = len(vocab)
    return vocab

def init_model(vocab_size):
    model = chainer.FunctionSet(
        embed=F.EmbedID(vocab_size, embed_units),
        hidden1=F.Linear(window * embed_units, hidden_units),
        output=F.Linear(hidden_units, label_num),
    )
    
    opt = optimizers.AdaGrad(lr=learning_rate)
    opt.setup(model)
    return model, opt

def make_label(sent):
    labels = list()
    pre_label = 1
    pre_char = ' '
    for char in sent:
        if not char == ' ':
            if pre_char == ' ':
                labels.append(0)
                pre_label = 0
            elif not pre_char == ' ':
                labels.append(1)
                pre_label = 1
        pre_char = char
    return labels

def train(char2id, model, optimizer):

    for epoch in range(n_epoch):
        print('epoch:', epoch)
        batch_count = 0
        accum_loss = 0
        for line in open(train_file):
            x = ''.join(line.strip().split())
            t = make_label(line.strip())
            for target in range(len(x)):
                label = t[target]
                pred, loss = forward_one(x, target, label)
                accum_loss += loss
            batch_count += 1
            if batch_count == batch_size:
                optimizer.zero_grads()
                accum_loss.backward()
                optimizer.update()
                accum_loss = 0
                batch_count = 0
     
        if not batch_count == 0:
            optimizer.zero_grads()
            accum_loss.backward()
            optimizer.update()
            accum_loss = 0
            batch_count = 0
        quick_test(char2id, model)


def quick_test(char2id, model):
    sent_cnt = 0
    labels = list()
    for line in open(train_file):
        x = ''.join(line.strip().split())
        t = make_label(line.strip())
        for target in range(len(x)):
            label = t[target]
            labels.append(label)
            loss, acc = forward_one(x, target, label)
        print ('predict sequence:', ''.join(label2seq(x,labels)))
        print ('true sequence***:', line.strip())
        labels = list()
        sent_cnt += 1
        if sent_cnt == 10:
            break

def test(char2id, model):
    labels = list()
    for line in open(train_file):
        x = ''.join(line.strip().split())
        t = make_label(line.strip())
        for target in range(len(x)):
            label = t[target]
            labels.append(label)
            loss, acc = forward_one(x, target, label)
        with open(result_raw, 'a') as test:
            test.write("{0}\n".format(''.join(label2seq(x,labels))))
            labels = list()

def forward_one(x, target, label):
    # make input window vector
    distance = window // 2
    char_vecs = list()
    x = list(x)
    for i in range(distance):
        x.append('</s>')
        x.insert(0,'<s>')
    for i in range(-distance, distance + 1):
        char = x[target + i]
        char_id = char2id[char]
        char_vec = model.embed(get_onehot(char_id))
        char_vecs.append(char_vec)

    concat = F.concat(tuple(char_vecs))
    hidden = model.hidden1(F.sigmoid(concat))
    pred = F.softmax(model.output(hidden))
    #pred = add_delta(pred)
    correct = get_onehot(label)
    return np.argmax(pred), F.softmax_cross_entropy(pred, correct)

def add_delta(p):
    return p + delta
def get_onehot(num):
    return chainer.Variable(np.array([num], dtype=np.int32))

def label2seq(x, labels):
    seq = list()
    for i in range(len(x)):
        if i == 0:
            seq.append(x[i])
        elif labels[i] == 0:
            seq.append(' ')
            seq.append(x[i])
        else:
            seq.append(x[i])
    return seq

if __name__ == '__main__':
    # reading config
    ini_file = sys.argv[1]
    ini = configparser.SafeConfigParser()
    ini.read(ini_file)
    train_file = ini.get('Data', 'train')
    test_file = ini.get('Data', 'test')
    result_raw = ini.get('Result', 'raw')
    config = ini.get('Result', 'config')
    evaluation = ini.get('Result', 'evaluation')
    window = int(ini.get('Parameters', 'window'))
    embed_units = int(ini.get('Parameters', 'embed_units'))
    hidden_units = int(ini.get('Parameters', 'hidden_units'))
    label_num = int(ini.get('Settings', 'label_num'))
    batch_size = int(ini.get('Settings', 'batch_size'))
    learning_rate = float(ini.get('Parameters', 'learning_rate'))
    n_epoch = int(ini.get('Settings', 'n_epoch'))
    delta = float(ini.get('Parameters', 'delta'))
    char2id = create_vocab()
    model, opt = init_model(len(char2id))
    train(char2id, model, opt)
    test(char2id, model)
