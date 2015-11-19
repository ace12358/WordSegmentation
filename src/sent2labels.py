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

def main():
    for line in open(sys.argv[1]):
        x = ''.join(line.strip().split())
        ts = make_label(line.strip())
        for t in ts:
            print(t)

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
    main()
