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
    print('####config####')
    for section in ini.sections():
        print ('[%s]' % (section))
        show_sectoin(ini, section)
    return

def show_sectoin(ini, section):
    '''
    show section
    '''
    for key in ini.options(section):
        show_key(ini, section, key)
    return

def show_key(ini, section, key):
    '''
    show key
    '''
    print ('%s.%s =%s' % (section, key, ini.get(section, key)))
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
        hidden1=F.Linear(window * embed_units + hidden_units, hidden_units),
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
    print('####Training####')
    for epoch in range(n_epoch):
        batch_count = 0
        accum_loss = 0
        line_cnt = 0
        hidden_vec = chainer.Variable(np.zeros((1, hidden_units),\
                                                     dtype=np.float32))
        for line in open(train_file):
            line_cnt += 1
            print("epoch: {0} trainig sentence: {1}".format(epoch,\
                                                 line_cnt), '\r', end='')
            x = ''.join(line.strip().split())
            t = make_label(line.strip())
            for target in range(len(x)):
                label = t[target]
                pred, loss = forward_one(x, target, label, hidden_vec)
                accum_loss += loss
            batch_count += 1
            if batch_count == batch_size:
                optimizer.zero_grads()
                accum_loss.backward()
                optimizer.weight_decay(lam)
                optimizer.update()
                accum_loss = 0
                batch_count = 0
            hidden_vec = chainer.Variable(np.zeros((1, hidden_units),\
                                                     dtype=np.float32))
     
        if not batch_count == 0:
            optimizer.zero_grads()
            accum_loss.backward()
            optimizer.update()
            accum_loss = 0
            batch_count = 0
        #quick_test(char2id, model)
    print('\nTraining Done!')


def quick_test(char2id, model):
    print('####quick test####')
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
        if sent_cnt == 3:
            break

def test(char2id, model):
    labels = list()
    with open(result_raw, 'w') as test:
        print("####Test####")
    line_cnt = 0
    hidden_vec = chainer.Variable(np.zeros((1, hidden_units),\
                                                     dtype=np.float32))
    for line in open(test_file):
        line_cnt += 1
        print("test sentence: {0}".format(line_cnt),'\r',end='')
        x = ''.join(line.strip().split())
        t = make_label(line.strip())
        for target in range(len(x)):
            label = t[target]
            labels.append(label)
            loss, acc = forward_one(x, target, label, hidden_vec)
        with open(result_raw, 'a') as test:
            test.write("{0}\n".format(''.join(label2seq(x,labels))))
            labels = list()
        hidden_vec = chainer.Variable(np.zeros((1, hidden_units),\
                                                     dtype=np.float32))
    print('\nTest Done!')


def forward_one(x, target, label, hidden_vec):
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
    concat2 = F.concat((concat, hidden_vec))
    hidden_vec = model.hidden1(F.sigmoid(concat2))
    pred = F.softmax(model.output(hidden_vec))
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
    lam = float(ini.get('Parameters', 'lam'))
    hidden_units = int(ini.get('Parameters', 'hidden_units'))
    label_num = int(ini.get('Settings', 'label_num'))
    batch_size = int(ini.get('Settings', 'batch_size'))
    learning_rate = float(ini.get('Parameters', 'learning_rate'))
    n_epoch = int(ini.get('Settings', 'n_epoch'))
    delta = float(ini.get('Parameters', 'delta'))
    show_config(ini)

    char2id = create_vocab()
    model, opt = init_model(len(char2id))
    train(char2id, model, opt)
    test(char2id, model)
