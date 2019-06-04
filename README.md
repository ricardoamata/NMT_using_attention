# NMT_using_attention
seq2seq model with attention layer, code implementation from [tensorflow](https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/contrib/eager/python/examples/nmt_with_attention/nmt_with_attention.ipynb)
for spanish to english translation. This is the initial version so there're some bugs to fix and things to improve.

## Training

to train use:

```
usage: training.py [-h] [-b BATCH_SIZE] [--emb_dim EMBEDDING_DIM]
                   [--units UNITS] [--dataset_size NUM_EXAMPLES]
                   [--epochs EPOCHS] [--seq_max_len SEQ_MAX_LEN]

Translator tester

optional arguments:
  -h, --help            show this help message and exit
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        size of the batch
  --emb_dim EMBEDDING_DIM
                        embedding dimension
  --units UNITS         internal size of the recurrent layers
  --dataset_size NUM_EXAMPLES
                        number of examples to train
  --epochs EPOCHS       number of epochs
  --seq_max_len SEQ_MAX_LEN
                        maximun length of the input sequence

```

## Translation

once you've trained for translating phrases use:

```
usage: translate.py [-h] [-p] in_seq

Translator tester

positional arguments:
  in_seq               sequence to translate

optional arguments:
  -h, --help           show this help message and exit
  -p, --plot_atention  plot attention grid

```


