import chainer
import numpy as np
import chainer.functions as F
import math
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
    best_score['0 <s>'] = 0.0 #best_score[index, pre_label]
    best_edge['0 <s>'] = None #best_edge[pre_label next_label]

    # process for </s>
    # process for <s>
    for next_label in range(label_num):
        score = best_score['0 <s>'] + 0.0 \
        + math.log(dists[0].data[0][next_label], 2)
        if not "%s %s" %(str(1), str(next_label)) in best_score \
        or score > best_score["%s %s" %(str(1), str(next_label))]:
            best_score["%s %s" %(str(1), str(next_label))] = score
            best_edge["%s %s" %(str(1), str(next_label))]\
            = "0 <s>"

    # process for </s>
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
        if not "%s %s" %(str(sent_length+1), str(pre_label)) in best_score\
        or score > best_score["%s </s>" %str(sent_length+1)]:
            best_score["%s </s>" %str(sent_length+1)] = score
            best_edge["%s </s>" %str(sent_length+1)]\
             = "%s %s" %(sent_length, pre_label)
    for i,j in best_edge.items():
        print(i,j)

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

    # make loss (difference between y_hat and y)
    diff_cnt = chainer.Variable(np.array([[0.0]], dtype=np.float32))  
    for i in range(len(labels)):
        if labels[i]!=ts[i]:
            diff_cnt += chainer.Variable(np.array([[1.0]], dtype=np.float32)) 
     
    predict_score = F.matmul(labels_matrix, dists_matrix, transb=True)+ T_labels
    true_score = F.matmul(ts_matrix, dists_matrix, transb=True) + T_ts
    
    score = predict_score - true_score + eta * diff_cnt
    print('predict_score:', predict_score.data)
    print('true_score:', true_score.data)
    print('loss:', eta * diff_cnt.data)
    return score
    
def train(char2id, model, optimizer):
    for epoch in range(n_epoch):
        print('epoch:', epoch )
        total_loss = 0.0
        batch_count = 0
        #accum_loss = 0
        accum_loss = chainer.Variable(np.array([[0.0]], dtype=np.float32))
        dists = list()
        for line in open(train_file):
            x = ''.join(line.strip().split())
            t = make_label(line.strip())
            for target in range(len(x)):
                label = t[target]
                dist = forward_one(x, target)
                dists.append(dist)
            y_hat = vitabi(dists)
            ############### debug
            print("############### debug")
            print("seq")
            for dist in dists:
                print(dist.data)
            print("trans")
            for pre in range(label_num):
                for cur in range(label_num):
                    print(pre, "to", cur, ": ", trans2para(pre, cur).data)
            print("vitabi: ", y_hat)
            ###############
            score = calc_score(y_hat, t, dists)
            if float(score.data) > 0:
                print('add loss')
                accum_loss += score
                total_loss += float(accum_loss.data)
            batch_count += 1
            dists = list()
            if batch_count == batch_size:
                optimizer.zero_grads()
                accum_loss.backward()
                optimizer.weight_decay(lam)
                optimizer.update()
                #accum_loss = 0
                accum_loss = chainer.Variable(np.array([[0.0]], dtype=np.float32))
                batch_count = 0
        # last update
        if not batch_count == 0:
            optimizer.zero_grads()
            accum_loss.backward()
            optimizer.weight_decay(lam)
            optimizer.update()
            #accum_loss = 0
            chainer.Variable(np.array([[0.0]], dtype=np.float32))
            batch_count = 0
        
        print ("input: %s" %x)
        print ("prediction***: %s" %''.join(label2seq(x, y_hat)))
        print ("true sequence: %s" %line.strip())

        print('total loss:', total_loss)
        print('accum loss:', accum_loss.data)

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
        dists = list()
        print ("input: %s" %x)
        print ("prediction***: %s" %''.join(label2seq(x, y_hat)))
        print ("true sequence: %s" %line.strip())

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
    pred = F.softmax(model.output(hidden))
    return pred

def get_onehot(num):
    return chainer.Variable(np.array([num], dtype=np.int32))
def decode():
    pass

if __name__ == '__main__':
    #config (hyper_parameter and so on)
    #train_file = '../data/jws_data/bccwj.train'
    #test_file = '../data/jws_data/bccwj.test'
    train_file = '../data/train.txt'
    test_file = '../data/test.txt'
    window = 3
    embed_units = 100
    hidden_units = 50
    label_num = 2
    batch_size = 30
    learning_rate = 0.05
    n_epoch = 10
    eta = 0.2
    lam = 0.1 ** 4
    #delta = 0.1 **10
    #about_1 = 0.9999999

    char2id = create_vocab()
    model, opt = init_model(len(char2id))
    train(char2id, model, opt)
    test(char2id, model, opt)
    
