import codecs
import time
import math
import sys
import argparse
import pickle
import copy
import os
import codecs
import chainer
import numpy as np
import chainer.functions as F
import chainer.links as L
from chainer import Link, Chain, ChainList, optimizers
from chainer import cuda, Variable, optimizers
import chainer.functions as F
from CharRNN import CharRNN, make_initial_state

def load_data(args):

    vocab = {}

    print ('%s/donald_trump_campaign_transcripts.txt'% args.data_dir)

    words = codecs.open('%s/donald_trump_campaign_transcripts.txt' % args.data_dir, 'rb', 'utf-8').read()

    words = list(words)

    dataset = np.ndarray((len(words),), dtype=np.int32)

    for i, word in enumerate(words):

        if word not in vocab:

            vocab[word] = len(vocab)

        dataset[i] = vocab[word]

    print ('corpus length:', len(words))

    print ('vocab size:', len(vocab))

    return dataset, words, vocab
	
# arguments

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir',                   type=str,   default='data')

parser.add_argument('--checkpoint_dir',             type=str,   default='cv')

parser.add_argument('--gpu',                        type=int,   default=-1)

parser.add_argument('--rnn_size',                   type=int,   default=128)

parser.add_argument('--learning_rate',              type=float, default=2e-3)

parser.add_argument('--learning_rate_decay',        type=float, default=0.97)

parser.add_argument('--learning_rate_decay_after',  type=int,   default=10)

parser.add_argument('--decay_rate',                 type=float, default=0.95)

parser.add_argument('--dropout',                    type=float, default=0.0)

parser.add_argument('--seq_length',                 type=int,   default=50)

parser.add_argument('--batchsize',                  type=int,   default=50)

parser.add_argument('--epochs',                     type=int,   default=50)

parser.add_argument('--grad_clip',                  type=int,   default=5)

parser.add_argument('--init_from',                  type=str,   default='')



args = parser.parse_args()

if not os.path.exists(args.checkpoint_dir):

    os.mkdir(args.checkpoint_dir)



n_epochs    = args.epochs

n_units     = args.rnn_size

batchsize   = args.batchsize

bprop_len   = args.seq_length

grad_clip   = args.grad_clip


train_data, words, vocab = load_data(args)

pickle.dump(vocab, open('%s/vocab.bin'%args.data_dir, 'wb'))



if len(args.init_from) > 0:

    model = pickle.load(open(args.init_from, 'rb'))

else:

    model = CharRNN(len(vocab), n_units)

if args.gpu >= 0:

    cuda.get_device(args.gpu).use()

    model.to_gpu()



optimizer = optimizers.RMSprop(lr=args.learning_rate, alpha=args.decay_rate, eps=1e-8)

optimizer.setup(model)

optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

whole_len    = train_data.shape[0]

jump         = whole_len // batchsize

epoch        = 0

start_at     = time.time()

cur_at       = start_at

state        = make_initial_state(n_units, batchsize=batchsize)

if args.gpu >= 0:

    accum_loss   = Variable(cuda.zeros(()))

    for key, value in state.items():

        value.data = cuda.to_gpu(value.data)

else:

    accum_loss   = Variable(np.zeros((), dtype=np.float32))



print ('going to train {} iterations'.format(jump * n_epochs))

for i in range(jump * n_epochs):

    x_batch = np.array([train_data[(jump * j + i) % whole_len]

                        for j in range(batchsize)])

    y_batch = np.array([train_data[(jump * j + i + 1) % whole_len]

                        for j in range(batchsize)])



    if args.gpu >=0:

        x_batch = cuda.to_gpu(x_batch)

        y_batch = cuda.to_gpu(y_batch)


    with chainer.using_config('train', True):
        state, loss_i = model.forward_one_step(x_batch, y_batch, state, dropout_ratio=args.dropout)

    accum_loss   += loss_i



    if (i + 1) % bprop_len == 0:  # Run truncated BPTT

        now = time.time()

        print ('{}/{}, train_loss = {}, time = {:.2f}'.format((i+1)/bprop_len, jump, accum_loss.data / bprop_len, now-cur_at))

        cur_at = now


        optimizer.target.cleargrads()
        accum_loss.backward()

        accum_loss.unchain_backward()  # truncate

        if args.gpu >= 0:

            accum_loss = Variable(cuda.zeros(()))

        else:

            accum_loss = Variable(np.zeros((), dtype=np.float32))


        optimizer.update()


    if (i + 1) % 10000 == 0:

        fn = ('%s/charrnn_epoch_%.2f.chainermodel' % (args.checkpoint_dir, float(i)/jump))

        pickle.dump(copy.deepcopy(model).to_cpu(), open(fn, 'wb'))

        pickle.dump(copy.deepcopy(model).to_cpu(), open('%s/latest.chainermodel'%(args.checkpoint_dir), 'wb'))



    if (i + 1) % jump == 0:

        epoch += 1



        if epoch >= args.learning_rate_decay_after:

            optimizer.lr *= args.learning_rate_decay

            print ('decayed learning rate by a factor {} to {}'.format(args.learning_rate_decay, optimizer.lr))



    sys.stdout.flush()