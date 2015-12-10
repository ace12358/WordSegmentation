#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Usage:
python ffnn_pointwise.txt config.ini
'''

import sys
import os
import chainer
import numpy as np
import math
import configparser
import chainer.functions as F
from chainer import optimizers


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
        trans=F.EmbedID(label_num, label_num),
        hidden1=F.Linear(window * embed_units, hidden_units),
        output=F.Linear(hidden_units, label_num),
    )
    #opt = optimizers.AdaGrad(lr=learning_rate)
    #opt = optimizers.SGD()
    opt = optimizers.Adam()
    opt.setup(model)
    return model, opt

def make_label(sent):
    labels = list()
    pre_char = ' '
    if label_num == 2: # BI
        for char in sent:
            if not char == ' ':
                if pre_char == ' ':
                    labels.append(0)
                elif not pre_char == ' ':
                    labels.append(1)
            pre_char = char
    elif label_num == 3: # B I S(single)
        for char in sent:
            if not char == ' ':
                if pre_char == ' ': # B or S
                    try:
                        if not sent[sent.find(char)+1] == ' ':
                            labels.append(0)
                        else:
                            labels.append(2)
                    except(IndexError):
                        labels.append(2)
                elif not pre_char == ' ': # I
                    labels.append(1)
            pre_char = char
    elif label_num == 4: # B M E S
        for char in sent:
            if not char == ' ':
                if pre_char == ' ': # B or S
                    try:
                        if not sent[sent.find(char)+1] == ' ':
                            labels.append(0)
                        else:
                            labels.append(3)
                    except(IndexError):
                        labels.append(3)
                elif not pre_char == ' ': # E or M
                    try:
                        if sent[sent.find(char)+1] == ' ':
                            labels.append(2)
                        else:
                            labels.append(1)
                    except(IndexError):
                        labels.append(2)
            pre_char = char
    return labels


def train(char2id, model, optimizer):
    print('####Training####')
    for epoch in range(n_epoch):
        batch_count = 0
        #accum_loss = 0
        accum_loss = chainer.Variable(np.array([[0.0]], dtype=np.float32))  
        line_cnt = 0
        for line in open(train_file):
            line_cnt += 1
            print("####epoch: {0} trainig sentence: {1}".format(epoch,\
                                                 line_cnt), '\r', end='')
            dists = list()
            x = ''.join(line.strip().split())
            gold_labels = make_label(line.strip())
            for target in range(len(x)):
                #label = t[target]
                dist = forward_one(x, target)
                dists.append(dist)

            #print("############### debug")
            #print("seq")
            #for dist in dists:
            #    print(dist.data)
            #print("trans")
            #for pre in range(label_num):
            #    for cur in range(label_num):
            #        print(pre, "to", cur, ": ", trans2para(pre, cur).data)
            # print("vitabi: ", y_hat)

                #accum_loss += loss
                #print('loss:',loss.data)
            #print('accum loss', accum_loss.data)
            y_hat = viterbi(dists)
            score = calc_score(y_hat, gold_labels, dists)
            batch_count += 1
            if float(score.data) > 0:
                accum_loss += score
            if batch_count == batch_size:
                optimizer.zero_grads()
                accum_loss.backward()
                optimizer.weight_decay(lam)
                optimizer.update()
                accum_loss = chainer.Variable(np.array([[0.0]], dtype=np.float32))  
                batch_count = 0
     
        if not batch_count == 0:
            optimizer.zero_grads()
            accum_loss.backward()
            optimizer.weight_decay(lam)
            optimizer.update()
            accum_loss = chainer.Variable(np.array([[0.0]], dtype=np.float32))  
            batch_count = 0
        epoch_test(char2id, model, epoch)
    print('\nTraining Done!')


def epoch_test(char2id, model, epoch):
    line_cnt = 0
    result_file = '{0}_{1}.txt'.format(result_raw.split('.txt')[0], epoch)
    for line in open(test_file):
        line_cnt += 1
        print('####epoch: {0} test and evaluation sentence: {1}####'\
                        .format(epoch,line_cnt), '\r', end = '')
        x = ''.join(line.strip().split())
        t = make_label(line.strip())
        dists = list()
        for target in range(len(x)):
            dist = forward_one(x, target)
            dists.append(dist)

        y_hat = viterbi(dists)
            #print('dist.data:', dist.data)
            #print('np.argmax(dist.data):', np.argmax(dist.data))

        with open(result_file, 'a') as test:
            test.write("{0}\n".format(''.join(label2seq(x, y_hat))))
    os.system('bash eval_japanese_ws.sh {0} {1} > temp'\
                                            .format(result_file, test_file))
    os.system('echo "####epoch{0} evaluation####" >> {1}'\
                                            .format(epoch, evaluation))
    os.system('cat temp >> {0}'.format(evaluation))
    os.system('rm temp')
        #print('predict sequence:', ''.join(label2seq(x,dists)))
        #print('true sequence***:', line.strip())

def forward_one(x, target):
    # make input window vector
    distance = window // 2
    char_vecs = list()
    x = list(x)
    for i in range(distance):
        x.append('</s>')
        x.insert(0,'<s>')
    for i in range(-distance+1 , distance + 2):
        char = x[target + i]
        char_id = char2id[char]
        char_vec = model.embed(get_onehot(char_id))
        char_vecs.append(char_vec)
    concat = F.concat(tuple(char_vecs))
    hidden = F.sigmoid(model.hidden1(concat))
    output = model.output(hidden)
    dist = F.softmax(output)
    #print(dist.data, label, np.argmax(dist.data))
    #correct = get_onehot(label)
    #print(output.data, correct.data)
    return dist

def get_onehot(num):
    return chainer.Variable(np.array([num], dtype=np.int32))

def label2seq(x, labels):
    seq = list()
    for i in range(len(x)):
        if label_num == 2:
            if i == 0:
                seq.append(x[i])
            elif labels[i] == 0:
                seq.append(' ')
                seq.append(x[i])
            else:
                seq.append(x[i])
        elif label_num == 3:
            if i == 0:
                seq.append(x[i])
            elif labels[i] == 0 or labels[i] == 2:
                seq.append(' ')
                seq.append(x[i])
            else:
                seq.append(x[i])
        elif label_num == 4:
            if i == 0:
                seq.append(x[i])
            elif labels[i] == 0 or labels[i] == 3:
                seq.append(' ')
                seq.append(x[i])
            else:
                seq.append(x[i])
    return seq

def viterbi(dists):
    # forward
    sent_length = len(dists)
    best_score = dict()
    best_edge = dict()
    best_score['0 <s>'] = 0.0 #best_score[index, pre_label]
    best_edge['0 <s>'] = None #best_edge[pre_label next_label]

    # process for <s>
    for next_label in range(label_num):
        score = best_score['0 <s>'] + 0.0 \
        + math.log(dists[0].data[0][next_label], 2)
        if not "%s %s" %(str(1), str(next_label)) in best_score \
        or score > best_score["%s %s" %(str(1), str(next_label))]:
            best_score["%s %s" %(str(1), str(next_label))] = score
            best_edge["%s %s" %(str(1), str(next_label))]\
            = "0 <s>"

    for i in range(1, sent_length):
        for pre_label in range(label_num):
            for next_label in range(label_num):
                if "%s %s" %(str(i),str(pre_label)) in best_score:
                    score = best_score["%s %s" %(str(i),str(pre_label))]\
                            + trans2para(pre_label, next_label).data[0][0]\
                            + math.log(dists[i].data[0][next_label],2)

                    if not "%s %s" %(str(i+1), str(next_label)) in best_score\
                    or score > best_score["%s %s" %(str(i+1), str(next_label))]:
                        best_score["%s %s" %(str(i+1), str(next_label))] = score
                        best_edge["%s %s" %(str(i+1), str(next_label))]\
                        = "%s %s" %(str(i), pre_label)

    # process for </s>
    for pre_label in range(label_num):
        score = best_score["%s %s" %(str(sent_length), pre_label)]
        #if not "%s %s" %(str(sent_length+1), str(pre_label)) in best_score\
        if not "%s %s" %(str(sent_length+1), '</s>') in best_score\
        or score > best_score["%s </s>" %str(sent_length+1)]:
            best_score["%s </s>" %str(sent_length+1)] = score
            best_edge["%s </s>" %str(sent_length+1)]\
             = "%s %s" %(sent_length, pre_label)
    #for i,j in best_edge.items():
    #    print(i,j)

    # backward
    labels = list()
    next_edge = best_edge["%s </s>" %str(sent_length+1)]
    while next_edge != "0 <s>":
    # add label to labels
        index = next_edge.split()[0] 
        label = next_edge.split()[1]
        labels.append(int(label))
        next_edge = best_edge[next_edge]
    labels.reverse()
    #print ('viterbi_labels:', labels)
    return labels

def trans2id(pre_label, next_label):
    return int(label_num) * int(pre_label) + int(next_label)

def trans2para(pre_label, next_label):
    pre_label = chainer.Variable(np.array([pre_label], dtype=np.int32))
    trans_vec = model.trans(pre_label)
    onehot = [1 if l == next_label else 0 for l in range(label_num)]
    onehot = chainer.Variable(np.array([onehot], dtype=np.float32))
    trans_para = F.matmul(onehot, F.softmax(trans_vec), transb=True)
    
    return trans_para

def get_trans(num):
    return chainer.Variable(np.array([num], dtype=np.int32))

def make_chainer_matrix(vector):
    return chainer.Variable(np.array([vector], dtype=np.float32))

def calc_score(labels, ts, dists):
    score = chainer.Variable(np.array([[0.0]], dtype=np.float32))
    T_labels = chainer.Variable(np.array([[0.0]], dtype=np.float32))
    T_ts = chainer.Variable(np.array([[0.0]], dtype=np.float32))
    # make labels vector and labels transitions vector
    labels_vec = list()
    pre_label = None
    for label in labels:
        if not pre_label == None:
            T_labels += trans2para(pre_label, label)
        for i in range(label_num):
            if i == label:
                labels_vec.append(1)
            else:
                labels_vec.append(0)
        pre_label = label
    labels_matrix = make_chainer_matrix(labels_vec)
    
    # make true labels vector
    ts_vec = list()
    pre_label = None
    for t in ts:
        if not pre_label == None:
            T_ts += trans2para(pre_label, t)
        for i in range(label_num):
            if i == t:
                ts_vec.append(1)
            else:
                ts_vec.append(0)
        pre_label = t
    ts_matrix = make_chainer_matrix(ts_vec)
    dists_matrix = F.concat(tuple(dists))

    #print('gold_labels:',ts_vec)
    #print('labels:', labels_vec)
    #print('labels_matrix.data[0]:',labels_matrix.data[0])
    #print(len(ts_matrix.data[0]))
    #print(len(dists_matrix.data[0]))

    # make loss (difference between y_hat and y)
    diff_cnt = chainer.Variable(np.array([[0.0]], dtype=np.float32))  
    for i in range(len(labels)):
        if labels[i]!=ts[i]:
            diff_cnt += chainer.Variable(np.array([[1.0]], dtype=np.float32))
            correct = get_onehot(ts[i])
            #print()
            #print(dists[i].data)
            #print(correct.data)
            #diff_cnt += F.softmax_cross_entropy(dists[i], correct)
     
    predict_score = F.matmul(labels_matrix, dists_matrix, transb=True)+ T_labels
    true_score = F.matmul(ts_matrix, dists_matrix, transb=True) + T_ts
    
    score = predict_score - true_score + eta * diff_cnt
    #print('predict_score:', predict_score.data)
    #print('true_score:', true_score.data)
    #print('loss:', eta * diff_cnt.data)
    return  score
 

if __name__ == '__main__':
    # reading config
    ini_file = sys.argv[1]
    ini = configparser.SafeConfigParser()
    ini.read(ini_file)
    train_file = ini.get('Data', 'train')
    test_file = ini.get('Data', 'test')
    result_raw = ini.get('Result', 'raw')
    config_file = ini.get('Result', 'config')
    evaluation = ini.get('Result', 'evaluation')
    window = int(ini.get('Parameters', 'window'))
    embed_units = int(ini.get('Parameters', 'embed_units'))
    hidden_units = int(ini.get('Parameters', 'hidden_units'))
    lam = float(ini.get('Parameters', 'lam'))
    label_num = int(ini.get('Settings', 'label_num'))
    batch_size = int(ini.get('Settings', 'batch_size'))
    learning_rate = float(ini.get('Parameters', 'learning_rate'))
    n_epoch = int(ini.get('Settings', 'n_epoch'))
    delta = float(ini.get('Parameters', 'delta'))
    eta = float(ini.get('Parameters', 'eta'))
    with open(config_file, 'w') as config:
        ini.write(config)

    char2id = create_vocab()
    model, opt = init_model(len(char2id))
    train(char2id, model, opt)
    #test(char2id, model)
