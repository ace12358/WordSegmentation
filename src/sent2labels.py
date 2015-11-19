#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Usage:
python sent2labels.txt [***.txt]
'''
import sys

def make_label(sent):
    B_cnt = 0
    I_cnt = 0 
    labels = list()
    pre_label = 1
    pre_char = ' '
    for char in sent:
        if not char == ' ':
            if pre_char == ' ':
                labels.append('B')
                B_cnt += 1
                pre_label = 0
            elif not pre_char == ' ':
                labels.append('I')
                I_cnt += 1
                pre_label = 1
        pre_char = char
    return labels, B_cnt, I_cnt

def main():
    B_cnt = 0
    I_cnt = 0 
    for line in open(sys.argv[1]):
        x = ''.join(line.strip().split())
        ts, b_cnt, i_cnt = make_label(line.strip())
        B_cnt += b_cnt
        I_cnt += i_cnt
        print(''.join(ts))
    print('B: {0}\nI: {1}'.format(B_cnt, I_cnt))

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
