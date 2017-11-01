import numpy as np
from chainer import Variable, Chain
import chainer.functions as F
import chainer.links as L

class CharRNN(Chain):
    def __init__(self, n_vocab, n_units):

        super(CharRNN, self).__init__(

            embed = L.EmbedID(n_vocab, n_units),

            l1_x = L.Linear(n_units, 4*n_units),

            l1_h = L.Linear(n_units, 4*n_units),

            l2_h = L.Linear(n_units, 4*n_units),

            l2_x = L.Linear(n_units, 4*n_units),

            l3   = L.Linear(n_units, n_vocab),

        )


    def forward_one_step(self, x_data, y_data, state, train=True, dropout_ratio=0.5):

        x = Variable(x_data)

        t = Variable(y_data)
		
        h0      = self.embed(x)

        h1_in   = self.l1_x(F.dropout(h0, ratio=dropout_ratio)) + self.l1_h(state['h1'])

        c1, h1  = F.lstm(state['c1'], h1_in)

        h2_in   = self.l2_x(F.dropout(h1, ratio=dropout_ratio)) + self.l2_h(state['h2'])

        c2, h2  = F.lstm(state['c2'], h2_in)

        y       = self.l3(F.dropout(h2, ratio=dropout_ratio))

        state   = {'c1': c1, 'h1': h1, 'c2': c2, 'h2': h2}

        if train:

            return state, F.softmax_cross_entropy(y, t)

        else:

            return state, F.softmax(y)



def make_initial_state(n_units, batchsize=50, train=True):

    return {name: Variable(np.zeros((batchsize, n_units), dtype=np.float32))

            for name in ('c1', 'h1', 'c2', 'h2')}