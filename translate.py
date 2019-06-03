import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from models import *
from preprocess import load_dataset, preprocess_sentence

def evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    
    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word2idx[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)
    
    result = ''

    hidden = [tf.zeros((1, encoder.enc_units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word2idx['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)
        
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.idx2word[predicted_id] + ' '

        if targ_lang.idx2word[predicted_id] == '<end>':
            return result, sentence, attention_plot
        
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')
    
    fontdict = {'fontsize': 14}
    
    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    plt.show()

def translate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ, graphic_mode):
    result, sentence, attention_plot = evaluate(sentence, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
        
    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(result))
    
    
    attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    if graphic_mode:
        plot_attention(attention_plot, sentence.split(' '), result.split(' '))

def main():
    tf.enable_eager_execution()

    parser = argparse.ArgumentParser(description='Translator tester')
    parser.add_argument("in_seq", type=str, help="sequence to translate")
    parser.add_argument("-p", "--plot_atention", dest="plot_attention", 
        help="plot attention grid", action="store_true")

    args = parser.parse_args()

    model_info = json.load(open('model_info.json', 'r', encoding='UTF-8'))

    encoder = Encoder(model_info['VIS'], model_info['ED'], model_info['UNITS'], model_info['BZ'])
    decoder = Decoder(model_info['VTS'], model_info['ED'], model_info['UNITS'], model_info['BZ'])

    input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_dataset(model_info['DATASET'], model_info['DSS'])

    evaluate("vuela", encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)

    encoder.load_weights('encoder')
    decoder.load_weights('decoder')

    
    translate(args.in_seq, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ, args.plot_attention)

if __name__ == "__main__":
    main()
    
