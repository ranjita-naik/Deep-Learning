* Python Version : 3.6
* Ipython Version : 4.0.1
* Framework : Chainer V2

#### I've ported https://github.com/yusuketomoto/chainer-char-rnn to chainer V2 and python 3.6. I am training the model on Donald Trump campaign transcripts.



#### Train


```
$ python train.py
```




#### Sampling

```
$ python sample.py \
--vocabulary data/vocab.bin \
--model cv/some_checkpoint.chainermodel \
--primetext donald --gpu -1
```