#!/usr/bin/env python
"""Sample script of recurrent neural network language model.

This code is ported from the following implementation written in Torch.
https://github.com/tomsercu/lstm

"""
from __future__ import division
from __future__ import print_function
import argparse

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import sru_link

# Definition of a recurrent net for language modeling
class RNNForLM(chainer.Chain):

    def __init__(self, n_vocab, n_units, rnn_type='lstm'):
        super(RNNForLM, self).__init__()
        self.rnn_type = rnn_type
        if self.rnn_type == 'sru':
            RNN1 = sru_link.SRU
            RNN2 = sru_link.SRU
        elif self.rnn_type == 'embedsru':
            RNN1 = sru_link.EmbedSRU
            RNN2 = sru_link.SRU
        elif self.rnn_type == 'superembedsru':
            RNN1 = sru_link.SuperEmbedSRU
            RNN2 = sru_link.SRU
        elif self.rnn_type == 'lstm':
            RNN1 = L.LSTM
            RNN2 = L.LSTM
        else:
            NotImplementedError()
        with self.init_scope():
            self.embed_x = L.EmbedID(n_vocab, n_units)
            self.l1 = RNN1(n_units, n_units)
            self.l2 = RNN2(n_units, n_units)
            self.l3 = L.Linear(n_units, n_vocab)

            if self.rnn_type in ['embedsru', 'superembedsru']:
                self.embed_y = L.EmbedID(n_vocab, n_units)
                self.embed_f = L.EmbedID(n_vocab, n_units)
                self.embed_r = L.EmbedID(n_vocab, n_units)

        for name, param in self.namedparams():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)

            if self.rnn_type in ['embedsru', 'superembedsru']:
                # gates should be mean of 0.5
                if 'embed_r' in name:
                    param.data[...] = F.sigmoid(param).data[...]
                if 'embed_f' in name:
                    param.data[...] = F.sigmoid(param).data[...]
            if self.rnn_type in ['superembedsru']:
                # stats should be same as vanilla values through sigmoid gates
                if 'embed_x' in name:
                    param.data[...] *= 0.5
                if 'embed_y' in name:
                    param.data[...] *= 0.5                    

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def __call__(self, x):
        h0 = self.embed_x(x)
        if self.rnn_type in ['embedsru', 'superembedsru']:
            h0 = F.dropout(h0)
            h0_y = self.embed_y(x)
            h0_y = F.dropout(h0_y)
            h0_f = self.embed_f(x)
            # no dropout due to the role as a gate
            h0_r = self.embed_r(x)
            # no dropout due to the role as a gate
            h1 = self.l1(h0, h0_y, h0_f, h0_r)
        else:
            h1 = self.l1(F.dropout(h0))
        h2 = self.l2(F.dropout(h1))
        y = self.l3(F.dropout(h2))
        return y


# Dataset iterator to create a batch of sequences at different positions.
# This iterator returns a pair of current words and the next words. Each
# example is a part of sequences starting from the different offsets
# equally spaced within the whole sequence.
class ParallelSequentialIterator(chainer.dataset.Iterator):

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size  # batch size
        # Number of completed sweeps over the dataset. In this case, it is
        # incremented if every word is visited at least once after the last
        # increment.
        self.epoch = 0
        # True if the epoch is incremented at the last iteration.
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        # Offsets maintain the position of each sequence in the mini-batch.
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        # NOTE: this is not a count of parameter updates. It is just a count of
        # calls of ``__next__``.
        self.iteration = 0
        # use -1 instead of None internally
        self._previous_epoch_detail = -1.

    def __next__(self):
        # This iterator returns a list representing a mini-batch. Each item
        # indicates a different position in the original sequence. Each item is
        # represented by a pair of two word IDs. The first word is at the
        # "current" position, while the second word at the next position.
        # At each iteration, the iteration count is incremented, which pushes
        # forward the "current" position.
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            # If not self.repeat, this iterator stops at the end of the first
            # epoch (i.e., when all words are visited once).
            raise StopIteration
        cur_words = self.get_words()
        self._previous_epoch_detail = self.epoch_detail
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        # Floating point version of epoch.
        return self.iteration * self.batch_size / len(self.dataset)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def get_words(self):
        # It returns a list of current words.
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        # It is important to serialize the state to be recovered on resume.
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.


# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
            train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        loss = 0
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            # Get the next batch (a list of tuples of two word IDs)
            batch = train_iter.__next__()

            # Concatenate the word IDs to matrices and send them to the device
            # self.converter does this job
            # (it is chainer.dataset.concat_examples by default)
            x, t = self.converter(batch, self.device)

            # Compute the loss at this time step and accumulate it
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters

        model = optimizer.target.predictor
        if model.rnn_type in ['embedsru', 'superembedsru']:
            for param in [model.embed_f.W, model.embed_r.W]:
                param.data[...] = model.xp.clip(param.data, 0.01, 0.99)

# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=20,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=35,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=39,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=10,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    parser.add_argument('--model', '-m', default='model.npz',
                        help='Model file name to serialize')
    parser.add_argument('--rnn-type', '-rnn',
                        default='lstm', choices=['lstm', 'sru', 'embedsru', 'superembedsru'])
    args = parser.parse_args()

    # Load the Penn Tree Bank long word sequence dataset
    train, val, test = chainer.datasets.get_ptb_words()
    n_vocab = max(train) + 1  # train is just an array of integers
    print('#vocab =', n_vocab)

    if args.test:
        train = train[:100]
        val = val[:100]
        test = test[:100]

    train_iter = ParallelSequentialIterator(train, args.batchsize)
    val_iter = ParallelSequentialIterator(val, 1, repeat=False)
    test_iter = ParallelSequentialIterator(test, 1, repeat=False)

    # Prepare an RNNLM model
    rnn = RNNForLM(n_vocab, args.unit, args.rnn_type)
    model = L.Classifier(rnn)
    model.compute_accuracy = False  # we only want the perplexity
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    decay_rate = 0.85
    trigger = (len(train) // (args.batchsize * args.bproplen), 'iteration')
    trainer.extend(extensions.observe_lr(), trigger=trigger)
    trainer.extend(extensions.ExponentialShift('lr', decay_rate, target=1e-6),
                   trigger=trigger)
    
    eval_model = model.copy()  # Model with shared params and distinct states
    eval_rnn = eval_model.predictor
    trainer.extend(extensions.Evaluator(
        val_iter, eval_model, device=args.gpu,
        # Reset the RNN state at the beginning of each evaluation
        eval_hook=lambda _: eval_rnn.reset_state()))

    interval = 10 if args.test else 500
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'iteration', 'perplexity', 'val_perplexity', 'lr']
    ), trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(
        update_interval=1 if args.test else 10))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.snapshot_object(
        model, 'model_iter_{.updater.iteration}'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()

    # Evaluate the final model
    print('test')
    eval_rnn.reset_state()
    evaluator = extensions.Evaluator(test_iter, eval_model, device=args.gpu)
    result = evaluator()
    print('test perplexity:', np.exp(float(result['main/loss'])))

    # Serialize the final model
    chainer.serializers.save_npz(args.model, model)


if __name__ == '__main__':
    main()
