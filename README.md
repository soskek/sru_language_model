# Recurrent Net Language Model using SRU

Experiments of RNN language models using SRU or other variants that I proposed.
SRU (Simple Recurrent Unit) is proposed by T.Lei and Y. Zhang in the paper ["Training RNNs as Fast as CNNs."](https://arxiv.org/pdf/1709.02755.pdf)
SRU implementation is by @unnonouno's [PR to Chainer](https://github.com/chainer/chainer/pull/3426).
The basic script is derived from [Chainer RNNLM example](https://github.com/chainer/chainer/tree/master/examples/ptb).

I also proposed two variants of SRU. The notations below follows equations (3)-(7) in the paper.

## 1. Embed-SRU
SRU takes `x` and produces `x^tilda`, `f` and `r` from `x` only. They do not depend on contexts, `c` and `h`. Thus, for computation efficiency, how about replacing them with new embeddings? In other words, `x`, `x^tilda`, `f` and `r` are independent word embeddings tied to a word at the time step.

## 2. Super-Embed-SRU
More extremely, terms `(1 - f) * x^tilda` and `(1 - r) * x` can be reduced to `x^tilda` and `x` without computation. Here, matrix product does not exist in this unit. The output is computed by two elementwise addition, two elementwise multiplication and a non-linear function.
```
c = f * c + x^tilda
h = r * tanh(c) + x
```

Note that even after training a vanilla SRU-based model, the model can be transformed into my Super-Embed-SRU without any loss of performance to reduce time complexity in return for increased space complexity.


# Results

This shows perplexity on test set of PennTreeBank language modeling dataset.

<table>
  <tr>
    <td></td>
    <td>LSTM</td>
    <td>SRU</td>
    <td>EmbedSRU</td>
    <td>SuperEmbedSRU</td>
  </tr>
  <tr>
    <td>Perplexity</td>
    <td>81.9</td>
    <td>86.0</td>
    <td>90.0</td>
    <td>87.5</td>
  </tr>
</table>

- SRU is worse than LSTM in the same number of layers.
- My proposal Super-Embed-SRU is promising, because the gap to the original SRU is small.


## Setting and notes
- Experimental setting is similar to one of the "Medium regularized LSTM" model of the paper, ["Recurrent Neural Network Regularization", Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals](https://arxiv.org/pdf/1409.2329v4.pdf).
    - (Perplexity is 86.2 (Dev) and 82.7 (Test) if tuning learning rate well; lr=1.0 during 1-6 epoch and then lr is decayed by 1/1.2 every epoch.)
- 2 layer RNNs with 650 units.
    - LSTM and LSTM, SRU and SRU, Embed-SRU and SRU, Super-Embed-SRU and SRU.
    - For experiments of Embed-SRU and Super-Embed-SRU, the 2nd layer is a vanilla SRU, because "embed" extension can be applied only for layers taking word embeddings.
- The main differences; (1) my learning rates are decayed every epoch by 0.85 with initial lr 1.0, and (2) threshold of gradient clipping is set to 10.
- My variants can be only in an RNN unit just after an embedding layer. For multi-layer RNN, other units are used for deeper layers.
- Of course, these variants do not equal to the vanilla SRU mainly for optimization. For example, connections of a shared `x` to `x^tilda`, `r` and `f` can be critical. This experiment aims for revealing the effect of such differences.
- In the vanilla SRU, gate vectors `r` and `f` are computed through sigmoid functions. In my variants, to follow the concept of "gate", I initialize embeddings `r` and `f` with mean 0.5 and limit their ranges to [0.01, 0.99] by clipping after every update.
- (Note: I did not optimize SRU for acceleration. This implementation does not contain multiplication across time step described in Section 2.3, while this is not related to task performance itself.)

Run
```
python train_ptb.py -g 0 -o out.lstm --rnn lstm
python train_ptb.py -g 0 -o out.sru --rnn sru
python train_ptb.py -g 0 -o out.embedsru --rnn embedsru
python train_ptb.py -g 0 -o out.superembedsru --rnn superembedsru
```