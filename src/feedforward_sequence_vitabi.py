import chainer
import numpy as np
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
        trans=F.EmbedID(label_num*label_num, 1),
        hidden1=F.Linear(window * embed_units, hidden_units),
        output=F.Linear(hidden_units, label_num),
    )
    
    opt = optimizers.AdaGrad(lr=learning_rate)
    opt.setup(model)
    return model, opt

def make_label(sent):
    labels = list()
    pre_label = None # impossible label
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


def vitabi(dists):
    # forward
    sent_length = len(dists)
    best_score = dict()
    best_edge = dict()
    # label_num stands for <s>
    # 0,1,...,label_num-1(3) stand for each label like [B,E,M,S]
    best_score['0 <s>'] = 0.0 #best_score[index, pre_label]
    best_edge['0 <s>'] = None #best_edge[pre_label next_label]
    # process for <s>
    for next_label in range(label_num):
        score = best_score['0 <s>'] + 0.0 + + dists[0].data[0][0]
        if not "%s %s" %(str(1), str(next_label)) in best_score \
        or score > best_score["%s %s" %(str(1), str(next_label))]:
            best_score["%s %s" %(str(1), str(next_label))] = score
            best_edge["%s %s" %(str(1), str(next_label))]\
            = "%s <s>" %str(0)
    for i in range(1, sent_length):
        for pre_label in range(label_num):
            for next_label in range(label_num):
                if "%s %s" %(str(i),str(pre_label)) in best_score:
                    score = best_score["%s %s" %(str(i),str(pre_label))]\
                            + trans2para(pre_label, next_label).data[0][0]\
                            + dists[i].data[0][0]
                    if not "%s %s" %(str(i+1), str(next_label)) in best_score\
                    or score > best_score["%s %s" %(str(i+1), str(next_label))]:
                        best_score["%s %s" %(str(i+1), str(next_label))] = score
                        best_edge["%s %s" %(str(i+1), str(next_label))]\
                        = "%s %s" %(str(i), pre_label)
    # process for </s>
    for pre_label in range(label_num):
        score = best_score["%s %s" %(str(sent_length), pre_label)]
        if not "%s %s" %(str(sent_length+1), str(pre_label)) in best_score\
        or score > best_score["%s </s>" %str(sent_length+1)]:
            best_score["%s </s>" %str(sent_length+1)] = score
            best_edge["%s </s>" %str(sent_length+1)]\
             = "%s %s" %(sent_length, pre_label)
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
    return labels

def trans2id(pre_label, next_label):
    return int(label_num) * int(pre_label) + int(next_label)

def trans2para(pre_label, next_label):
    trans_id = trans2id(pre_label, next_label)
    trans_para = model.trans(get_trans(trans_id))
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
            if str(i) == label:
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
            if str(i) == t:
                ts_vec.append(1)
            else:
                ts_vec.append(0)
            pre_label = t
    ts_matrix = make_chainer_matrix(ts_vec)

    # make dists_vector
    # dists_vec = list()
    # for dist in dists:
    #     for p in dist.data:
    #         dists_vec.append(p)
    # dists_matrix =  make_chainer_matrix(dists_vec)  
    dists_matrix = F.concat(tuple(dists))

    # make loss (difference between y_hat and y)
    diff_cnt = chainer.Variable(np.array([[0.0]], dtype=np.float32))  
    for i in range(len(labels)):
        if labels[i]!=ts[i]:
            diff_cnt += chainer.Variable(np.array([[1.0]], dtype=np.float32)) 
    
    score = F.matmul(labels_matrix, dists_matrix, transb=True)+ T_labels \
            - F.matmul(ts_matrix, dists_matrix, transb=True) - T_ts  \
            + eta * diff_cnt
    return score
  
    
def train(char2id, model, optimizer):

    for epoch in range(n_epoch): 
        batch_count = 0
        accum_loss = 0
        dists = list()
        for line in open(train_file):
            x = ''.join(line.strip().split())
            t = make_label(line.strip())
            for target in range(len(x)):
                label = t[target]
                dist = forward_one(x, target)
                dists.append(dist)
            y_hat = vitabi(dists)
            score = calc_score(y_hat, t, dists)
            accum_loss += score
            batch_count += 1
            dists = list()
            if batch_count == batch_size:
                optimizer.zero_grads()
                accum_loss.backward()
                optimizer.weight_decay(lam)
                optimizer.update()
                accum_loss = 0
                batch_count = 0
 
        if not batch_count == 0:
            optimizer.zero_grads()
            accum_loss.backward()
            optimizer.weight_decay(lam)
            optimizer.update()
            accum_loss = 0
            batch_count = 0

def test(char2id, model, optimizer):
    dists = list()
    for line in open(test_file):
        x = ''.join(line.strip().split())
        t = make_label(line.strip())
        for target in range(len(x)):
            label = t[target]
            dist = forward_one(x, target)
            dists.append(dist)
        y_hat = vitabi(dists)
        print (x)
        print (y_hat)
        print (t)
"""
    sum_accuracy = 0
    sum_loss = 0
    
    batch_count = 0
    accum_loss = 0
    labels = list()
    for line in open(train_file):
        x = ''.join(line.strip().split())
        t = make_label(line.strip())
        for target in range(len(x)):
            label = t[target]
            labels.append(label)
            loss, acc = forward_one(x, target, label)
        print (x)
        print (labels)
        print (t)
    #sum_loss += float(loss.data) * len(y)
    #sum_accuracy += float(acc.data) * len(y)

#print('test  mean loss={}, accuracy={}'.format(
#    sum_loss / N_test, sum_accuracy / N_test))
"""
def forward_one(x, target):
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
    pred = model.output(hidden)
    return pred

def get_onehot(num):
    return chainer.Variable(np.array([num], dtype=np.int32))
def decode():
    pass

if __name__ == '__main__':
    #config (hyper_parameter and so on)
    train_file = '../data/train.txt'
    test_file = '../data/test.txt'
    window = 3
    embed_units = 100
    hidden_units = 50
    label_num = 2
    batch_size = 30
    learning_rate = 0.1
    n_epoch = 10
    eta = 0.5
    lam = 0.1 ** 4

    char2id = create_vocab()
    model, opt = init_model(len(char2id))
    train(char2id, model, opt)
    test(char2id, model, opt)
    
