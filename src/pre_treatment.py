#!/Users/yukitomo/.pyenv/versions/chainer_py343/bin/python
#-*-coding:utf-8-*-
#2015-10-20 Yuki Tomo

"""
How to Use
$ python3 pre_treatment.py ws_ref ws_out > char_ref_out.txt
$ perl conlleval.pl -d “\t” a_ref_out.txt
"""

import sys
from argparse import ArgumentParser

def label_chars(sent):
    chars_labels = []
    label = "B"
    for char in sent:
        if char == " ":
            label = "B"
        else:
            chars_labels.append((char,label))
            label = "I"
    return chars_labels

def treatment(ref_file, rslt_file):
 
    with open(ref_file) as ref, open(rslt_file) as rslt:
        for (i, sent_ref) , sent_rslt in  zip(enumerate(ref), rslt):
            #split_phrases
            chars_ref = label_chars(sent_ref.strip())
            chars_rslt = label_chars(sent_rslt.strip())

            if len(chars_ref) != len(chars_rslt):
                print(i)
                print("ref_sent  :", sent_ref)
                print("rslt_sent :", sent_rslt)  
                raise ValueError("ref and rslt doesn't match about characters")

        
            for char_ref, char_rslt in zip(chars_ref, chars_rslt):
                print(char_ref[0], char_ref[1], char_rslt[1],sep="\t")
                #print("a", char_ref[1], char_rslt[1],sep="\t")

            print("")

def parse_args():

    p = ArgumentParser(description='Word segmentation using LSTM-RNN')

    p.add_argument('ref', help='reference file')
    p.add_argument('rslt', help='system result')

    args = p.parse_args()

    # check args
    try:
        if args.ref == "": raise ValueError('you must set ref = reference')
        if args.rslt == "": raise ValueError('you must set rslt = system result')

    except Exception as ex:
        p.print_usage(file=sys.stderr)
        print(ex, file=sys.stderr)
        sys.exit()

    return args

def main():
    args = parse_args()

    treatment(args.ref, args.rslt) 

if __name__ == '__main__':
    main()
