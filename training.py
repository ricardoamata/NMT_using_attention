import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import time
import os
import json
import argparse
from models import Encoder, Decoder
from preprocess import *

def max_length(tensor):
    return max(len(t) for t in tensor)

def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)

def main():
    parser = argparse.ArgumentParser(description='Translator tester')
    parser.add_argument("-b", "--batch_size", dest="batch_size",
        help="size of the batch", type=int, default=64)
    parser.add_argument("--emb_dim", dest="embedding_dim",
        help="embedding dimension", type=int, default=256)
    parser.add_argument("--units", dest="units", 
        help="internal size of the recurrent layers", type=int, default=1024)
    parser.add_argument("--dataset_size", dest="num_examples", 
        help="number of examples to train", type=int, default=30000)
    parser.add_argument("--epochs", dest="epochs", 
        help="number of epochs", type=int, default=10)
    parser.add_argument("--seq_max_len", dest="seq_max_len", 
        help="maximun length of the input sequence", type=int, default=500)

    args = parser.parse_args()

    tf.enable_eager_execution()

    path_to_zip = tf.keras.utils.get_file(
        'spa-eng.zip', origin='http://download.tensorflow.org/data/spa-eng.zip', 
        extract=True
    )
    path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"

    num_examples = args.num_examples
    input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(path_to_file, num_examples)

    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = args.batch_size
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    embedding_dim = args.embedding_dim
    units = args.units
    vocab_inp_size = len(inp_lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

    optimizer = tf.train.AdamOptimizer()

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    optimizer = tf.train.AdamOptimizer()

    EPOCHS = args.epochs

    for epoch in range(EPOCHS):
        start = time.time()
        
        hidden = encoder.initialize_hidden_state()
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0
            
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)
                dec_hidden = enc_hidden
                dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)       
                
                for t in range(1, targ.shape[1]):
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                    loss += loss_function(targ[:, t], predictions)
                    dec_input = tf.expand_dims(targ[:, t], 1)
            
            batch_loss = (loss / int(targ.shape[1]))
            total_loss += batch_loss
            variables = encoder.variables + decoder.variables
            gradients = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(gradients, variables))
            
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        if (epoch + 1) % 2 == 0:
          checkpoint.save(file_prefix = checkpoint_prefix)
        
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / N_BATCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    encoder.save_weights('encoder', save_format='tf')
    decoder.save_weights('decoder', save_format='tf')

    model_info = {
        'VIS': vocab_inp_size, 
        'VTS': vocab_tar_size, 
        'ED': embedding_dim, 
        'UNITS': units, 
        'BZ': BATCH_SIZE,
        'DATASET': path_to_file,
        'DSS': num_examples
    }
    
    with open('model_info.json', 'w') as outfile:
        json.dump(model_info, outfile)

if __name__ == "__main__":
    main()
    
