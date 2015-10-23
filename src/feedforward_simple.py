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
        hidden1=F.Linear(window * embed_units, hidden_units),
        output=F.Linear(hidden_units, label_num),
    )
    
    opt = optimizers.AdaGrad(lr=learning_rate)
    opt.setup(model)
    return model, opt

def make_label(sent):
    labels = list()
    for char in sent:
        if not char == ' ':
            labels.append(1)
    return labels

def train(char2id, model, optimizer):

    for epoch in range(n_epoch): 
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

def test(char2id, model, optimizer):
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
    pred = model.output(hidden)
    correct = get_onehot(label)
    return np.argmax(pred), F.softmax_cross_entropy(pred, correct)

def get_onehot(num):
    #return chainer.Variable(np.array([[num]], dtype=np.int32))
    return chainer.Variable(np.array([num], dtype=np.int32))
def decode():
    pass

if __name__ == '__main__':
    train_file = '../data/train.txt'
    test_file = '../data/test.txt'
    window = 3
    embed_units = 100
    hidden_units = 50
    label_num = 4
    batch_size = 30
    learning_rate = 0.1
    n_epoch = 10

    char2id = create_vocab()
    model, opt = init_model(len(char2id))
    train(char2id, model, opt)
    test(char2id, model, opt)
    
