from chainer import cuda
import chainer.functions as F
from chainer import link
from chainer.links.connection import linear

import sru_func

class SRU(link.Chain):

    def __init__(self, in_size, out_size=None):
        super(SRU, self).__init__()
        self.activation = F.tanh
        with self.init_scope():
            self.W = linear.Linear(in_size, out_size * 3)
        self.c = None

    def reset_state(self):
        self.c = None

    def __call__(self, x):
        u = self.W(x)
        if self.c is None:
            c = cuda.cupy.zeros(x.shape, 'f')
        else:
            c = self.c
        self.c, h = sru_func.sru(c, x, u)
        return h

def embed_sru(c, x, y, f, r, activation=F.tanh):
    c = f * c + (1. - f) * y
    h = r * activation(c) + (1. - r) * x
    return c, h

def super_embed_sru(c, x, y, f, r, activation=F.tanh):
    c = f * c + y
    h = r * activation(c) + x
    return c, h

class EmbedSRU(link.Chain):

    def __init__(self, in_size, out_size=None):
        super(EmbedSRU, self).__init__()
        self.activation = F.tanh
        self.c = None
        self.super_embed = False

    def reset_state(self):
        self.c = None

    def __call__(self, x, y, f, r):
        if self.c is None:
            c = cuda.cupy.zeros(x.shape, 'f')
        else:
            c = self.c

        if self.super_embed:
            self.c, h = super_embed_sru(c, x, y, f, r, self.activation)
        else:
            self.c, h = embed_sru(c, x, y, f, r, self.activation)
        return h

class SuperEmbedSRU(EmbedSRU):
    def __init__(self, in_size, out_size=None):
        super(SuperEmbedSRU, self).__init__(in_size, out_size)
        self.super_embed = True
