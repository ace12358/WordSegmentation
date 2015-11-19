#!/usr/bin/python3

#import my_settings

import sys
import math
import numpy as np
import time
from argparse import ArgumentParser

from chainer import functions, optimizers

import util.generators as gens
from util.functions import trace, fill_batch
from util.model_file import ModelFile
from util.vocabulary import Vocabulary

from util.chainer_cpu_wrapper import wrapper
#from util.chainer_gpu_wrapper import wrapper


class TransSegmentationModel:
    def __init__(self):
        pass

    def __make_model(self):
        self.__model = wrapper.make_model(
            w_xh = functions.EmbedID(2 * self.__n_context * len(self.__vocab), self.__n_hidden),
            w_hy = functions.Linear(self.__n_hidden, self.__n_labels),
            trans = functions.EmbedID(self.__n_labels * self.__n_labels, 1), #各ラベル(0,1)間の遷移のweight #確率としておく softmaxかます
        )

    @staticmethod
    def new(vocab, n_context, n_hidden, n_labels, eta):
        self = TransSegmentationModel()
        self.__vocab = vocab
        self.__n_context = n_context
        self.__n_hidden = n_hidden
        self.__n_labels = n_labels
        self.__eta = eta
        self.__make_model()
        return self

    def save(self, filename):
        with ModelFile(filename, 'w') as fp:
            self.__vocab.save(fp.get_file_pointer())
            fp.write(self.__n_context)
            fp.write(self.__n_hidden)
            wrapper.begin_model_access(self.__model)
            fp.write_embed(self.__model.w_xh)
            fp.write_linear(self.__model.w_hy)
            wrapper.end_model_access(self.__model)

    @staticmethod
    def load(filename):
        self = TransSegmentationModel()
        with ModelFile(filename) as fp:
            self.__vocab = Vocabulary.load(fp.get_file_pointer())
            self.__n_context = int(fp.read())
            self.__n_hidden = int(fp.read())
            self.__make_model()
            wrapper.begin_model_access(self.__model)
            fp.read_embed(self.__model.w_xh)
            fp.read_linear(self.__model.w_hy)
            wrapper.end_model_access(self.__model)
        return self

    def init_optimizer(self):
        self.__opt = optimizers.AdaGrad(lr=0.01)
        self.__opt.setup(self.__model)


    # B, I = [0, 1]
    def __make_input(self, is_training, text):
        c = self.__vocab.stoi
        k = self.__n_context - 1
        word_list = text.split()
        letters = [c('<s>')] * k + [c(x) for x in ''.join(word_list)] + [c('</s>')] * k #１文字目、最後の文字を補間するもの
        if is_training:
            labels = []
            for x in word_list:
                labels += [0] + [1] * (len(x) - 1)  # -1:単語中 1:単語くぎれる
            return letters, labels[1:]
        else:
            return letters, None

    
    def __forward(self, is_training, text):
        m = self.__model
        tanh = functions.tanh
        softmax = functions.softmax
        letters, labels = self.__make_input(is_training, text) #文字列(スペースなし), 単語境界ラベル(-1,+1)
        dists = []
        accum_loss = wrapper.make_var([[0.0]]) if is_training else None
        """
        debug
        print("length, letters : ",len(letters), letters)
        print("len(self.__vocab) : ",len(self.__vocab))
        """
            
        for n in range(len(letters) - 2 * self.__n_context + 1):
            """
            debug
            print("n : ",n)
            """
            s_hu = wrapper.zeros((1, self.__n_hidden))
            
            for k in range(2 * self.__n_context):
                wid = k * len(self.__vocab) + letters[n + k]
                s_x = wrapper.make_var([wid], dtype=np.int32)
                s_hu += m.w_xh(s_x)
                """
                debug
                print("k : ",k)
                print("wid : ",wid)
                """  

            
            s_hv = tanh(s_hu)
            s_y = softmax(m.w_hy(s_hv))

            """
            debug
            print("s_y : ",s_y.data) 
            """

            dists.append(s_y)
            
        
        """
        debug
        for i,dist in enumerate(dists):
            print("dist_", i,": ", wrapper.get_data(dist))
        self.check_trans()
        """

        
        sys_ys, end_score = self.viterbi(dists)

        """
        debug
        print("end_score : ", end_score)
        """
        
        if is_training:
            """
            debug
            print("labels : ", labels)
            print("sys_ys : ", sys_ys)
            """

            loss = self.calc_loss(sys_ys, labels, dists)

            """
            debug
            print("loss : ", wrapper.get_data(loss))
            """

            if float(wrapper.get_data(loss)) > 0:
                accum_loss += loss

        return sys_ys, accum_loss

    def calc_loss(self, sys_ys, ref_ys, dists):
        loss = wrapper.make_var([[0.0]])
        sys_Tscore = wrapper.make_var([[0.0]])
        ref_Tscore = wrapper.make_var([[0.0]])
        
        sys_Tscore, sys_vecs = self.calc_trans_score(sys_ys) #chainer.Variable, 1hotvecがconcateされたもの
        sys_matrix = wrapper.make_var([sys_vecs])

        ref_Tscore, ref_vecs = self.calc_trans_score(ref_ys) #chainer.Variable, 1hotvec
        ref_matrix = wrapper.make_var([ref_vecs])

        dists_matrix = functions.concat(tuple(dists))

        #異なるラベル数のカウント
        diff_cnt = wrapper.make_var([[0.0]])
        for sys_y, ref_y in zip(sys_ys, ref_ys):
            if sys_y != ref_y:
                diff_cnt += wrapper.make_var([[1.0]])

        #max 0
        loss = functions.matmul(sys_matrix, dists_matrix, transb=True) + sys_Tscore\
               - functions.matmul(ref_matrix, dists_matrix, transb=True) - ref_Tscore\
               + self.__eta * diff_cnt

        """
        debug
        print("sys_score trans : ", wrapper.get_data(functions.matmul(sys_matrix, dists_matrix, transb=True)), wrapper.get_data(sys_Tscore))
        print("ref_score trans : ", wrapper.get_data(functions.matmul(ref_matrix, dists_matrix, transb=True)), wrapper.get_data(ref_Tscore))
        print("diff_cnt penal : ",wrapper.get_data(diff_cnt), wrapper.get_data(self.__eta * diff_cnt))
        """

        return loss

    def calc_trans_score(self, labels):
        trans_score = wrapper.make_var([[0.0]])
        labels_vec = list()
        pre_label = None

        for label in labels:
            if pre_label != None:
                """
                debug
                print("pre_label : ",pre_label)
                print("label : ",label)
                """
                trans_score += wrapper.get_data(self.get_trans_prob(pre_label))[0][label] #trans_score(pre_label → 0,1)をsoftmaxをかけてスコアリング
            #1hot CONCATE vector
            for i in range(self.__n_labels):
                if i == label:
                    labels_vec.append(1)
                else:
                    labels_vec.append(0)
            pre_label = label

        return trans_score, labels_vec


    def viterbi(self, dists):
        length = len(dists)
        best_score = dict()
        best_edge = dict()
        trans_prob = list()
        """
        trans_prob  0  :  [[ 0.47865251  0.52134752]]
        trans_prob  1  :  [[ 0.89475453  0.10524546]]
        """

        #trans_score をsoftmaxかけたものを予め計算
        for pre_label in range(self.__n_labels):
            trans_prob.append(self.get_trans_prob(pre_label))

        """
        debug
        for pre_label in range(self.__n_labels):
            print("trans_prob ", pre_label ," : ",wrapper.get_data(trans_prob[pre_label]))
        """

        #best_score["0 0"] = float('inf') #index:0, label:B からスタート
        best_score["0 0"] = 1000
        best_edge["0 0"] = None 

        #forward
        for i in range(length):
            if i == 0:
                n_labels = 1 #最初はB-I or B-B のみ
            else:
                n_labels = self.__n_labels

            for pre_label in range(n_labels):
                for next_label in range(self.__n_labels):                    
                    score = self.node_score(i, pre_label, next_label, dists, trans_prob, best_score)
                    next_index_label = "%s %s" %(i+1, str(next_label))

                    """
                    debug
                    print("best_score : ", best_score)
                    print("pre_label next_label : ", pre_label, next_label)
                    print("score : ", score)
                    """


                    if not next_index_label in best_score or score < best_score[next_index_label]:
                        best_score[next_index_label] = score
                        best_edge[next_index_label] = "%s %s" %(i, pre_label)                

        #for </s>
        for pre_label in range(self.__n_labels):
            score = best_score["%s %s" %(length, pre_label)]
            next_index_label = "%s </s>" %(length+1)

            if not next_index_label in best_score or score < best_score[next_index_label]:
                best_score[next_index_label] = score
                best_edge[next_index_label] = "%s %s" %(length, pre_label)                

        #最終的なスコア
        end_score = best_score[next_index_label]
        """
        debug
        print("best_edge : ",best_edge)
        """

        #backword
        labels = list()
        next_edge = best_edge["%s </s>" %(length+1)]

        while next_edge != "0 0":
            index = next_edge.split()[0] 
            label = next_edge.split()[1] # 0 or 1   
            labels.append(int(label))
            next_edge = best_edge[next_edge]
        labels.reverse()
        return labels, end_score

    def node_score(self, index, pre_label, next_label, dists, trans_prob, best_score):

        score = best_score["%s %s" %(index,str(pre_label))]\
        + trans_prob[pre_label].data[0][next_label]\
        - math.log(dists[index].data[0][next_label],2)
        return score

    def train(self, text):
        self.__opt.zero_grads()
        labels, accum_loss = self.__forward(True, text)
        accum_loss_f = float(wrapper.get_data(accum_loss)) 
        accum_loss.backward()
        self.__opt.weight_decay(1)
        self.__opt.clip_grads(5)
        self.__opt.update()
        return labels, accum_loss_f

    def predict(self, text):
        return self.__forward(False, text)[0]

    def get_trans(self, pre_label, next_label):
        trans_id = int(self.__n_labels) * int(pre_label) + int(next_label)
        trans_val = self.__model.trans(wrapper.make_var_int([trans_id])) 
        return trans_val

    def get_trans_prob(self, pre_label):
        trans_scores = []
        for next_label in range(self.__n_labels):
            trans_scores.append(self.get_trans(pre_label, next_label))
        scores = functions.softmax(functions.concat(trans_scores))
        return scores

    def check_trans(self):
        print("pre next transition_score")
        for pre_label in range(self.__n_labels):
            for next_label in range(self.__n_labels):
                trans_val = self.get_trans(pre_label, next_label)
                print(pre_label, next_label, wrapper.get_data(trans_val))


def parse_args():
    def_vocab = 2500
    def_hidden = 100
    def_epoch = 100
    def_context = 3
    def_labels = 2 #B,I
    def_eta = 10

    p = ArgumentParser(description='Word segmentation using feedforward neural network')

    p.add_argument('mode', help='\'train\' or \'test\'')
    p.add_argument('corpus', help='[in] source corpus')
    p.add_argument('model', help='[in/out] model file')
    p.add_argument('--vocab', default=def_vocab, metavar='INT', type=int,
        help='vocabulary size (default: %d)' % def_vocab)
    p.add_argument('--hidden', default=def_hidden, metavar='INT', type=int,
        help='hidden layer size (default: %d)' % def_hidden)
    p.add_argument('--epoch', default=def_epoch, metavar='INT', type=int,
        help='number of training epoch (default: %d)' % def_epoch)
    p.add_argument('--context', default=def_context, metavar='INT', type=int,
        help='width of context window (default: %d)' % def_context)
    p.add_argument('--labels', default=def_labels, metavar='INT', type=int,
        help='number of labels (default: %d)' % def_labels)
    p.add_argument('--eta', default=def_eta, metavar='INT', type=int,
        help='value of eta (default: %d)' % def_eta)

    args = p.parse_args()

    # check args
    try:
        if args.mode not in ['train', 'test']: raise ValueError('you must set mode = \'train\' or \'test\'')
        if args.vocab < 1: raise ValueError('you must set --vocab >= 1')
        if args.hidden < 1: raise ValueError('you must set --hidden >= 1')
        if args.epoch < 1: raise ValueError('you must set --epoch >= 1')
        if args.context < 1: raise ValueError('you must set --context >= 1')
        if args.labels < 2: raise ValueError('you must set --labels >= 2')
        if args.labels < 1: raise ValueError('you must set --eta < 1')
    except Exception as ex:
        p.print_usage(file=sys.stderr)
        print(ex, file=sys.stderr)
        sys.exit()

    return args


def make_hyp(letters, labels): #dists = [[[ 0.34384966  0.65615034]], [[ 0.78964823  0.21035172]],...]
    """
    debug
    print("make_hyp")
    print("letters", letters)
    print("labels", labels)
    """
    hyp = letters[0] #1文字目は前にspace無し
    for w, label in zip(letters[1:], labels):
        """debug
        print("w label : ", w, label)
        """
        if label == 0:
            hyp += ' '
        hyp += w
    return hyp


def train_model(args):
    train_begin = time.time()
    trace('making vocaburaries ...')
    vocab = Vocabulary.new(gens.letter_list(args.corpus), args.vocab) 

    trace('begin training ...')
    model = TransSegmentationModel.new(vocab, args.context, args.hidden, args.labels, args.eta)

    for epoch in range(args.epoch):
        epoch_beg = time.time() 
        trace('START epoch %d/%d: ' % (epoch + 1, args.epoch))
        trained = 0
        total_loss = 0

        model.init_optimizer()

        with open(args.corpus) as fp:
            for text in fp:
                word_list = text.split()
                if not word_list:
                    continue

                text = ' '.join(word_list)
                letters = ''.join(word_list)
                labels, accum_loss_f = model.train(text)
                total_loss += accum_loss_f
                trained += 1
                hyp = make_hyp(letters, labels)
                
                """for 1sentence output
                trace("accum_loss : %lf"% (accum_loss_f))
                trace('epoch %d/%d: ' % (epoch + 1, args.epoch))
                trace('trained %d: '% trained)
                trace(text)
                trace(hyp)
                """
                """
                if trained % 100 == 0:
                    trace('  %8d' % trained)
                """
        trace('FINISHED epoch %d/%d: ' % (epoch + 1, args.epoch))
        trace('total_loss : %lf'%total_loss)
        trace('saving model ...')
        model.save(args.model + '.%03d' % (epoch + 1))
        epoch_time = time.time() - epoch_beg
        trace('elapsed_time/1epoch : %lf'%epoch_time)

    trace('finished.')
    elapsed_time = time.time() - train_begin
    trace('train_time : %lf'%elapsed_time)
    trace('')

def test_model(args):
    trace('loading model ...')
    model = TransSegmentationModel.load(args.model)
    
    trace('generating output ...')

    with open(args.corpus) as fp:
        for text in fp:
            letters = ''.join(text.split())
            if not letters:
                print()
                continue
            scores = model.predict(text)
            hyp = make_hyp(letters, scores)
            print(hyp)

    trace('finished.')


def main():
    args = parse_args()

    trace('initializing CUDA ...')
    wrapper.init()

    if args.mode == 'train': train_model(args)
    elif args.mode == 'test': test_model(args)


if __name__ == '__main__':
    main()

