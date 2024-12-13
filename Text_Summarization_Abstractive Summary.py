#!/usr/bin/env python
# coding: utf-8

# # Text Processing - Abstractive Summarization

# ### Import Libraries

# In[1]:


import os
import re
import pandas as pd
import numpy as np
from contractions import contractions_dict
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, Bidirectional, TimeDistributed, Lambda, Attention
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge


# ### Custom Methods 

# In[2]:


def data_preprocess(df1,df2):
    df1Cols = df1.columns.tolist()
    df1Cols.remove('headlines')
    df1Cols.remove('text')
    df1.drop(df1Cols, axis='columns', inplace=True)

    df = pd.concat([df1, df2], axis='rows')
    del df1, df2
    df = df.sample(frac=1).reset_index(drop=True)
    df.text = df.text.apply(str.lower)
    df.headlines = df.headlines.apply(str.lower)
    return df


def expand_contractions(text, contraction_map=contractions_dict):
    # Using regex for getting all contracted words
    contractions_keys = '|'.join(contraction_map.keys())
    contractions_pattern = re.compile(f'({contractions_keys})', flags=re.DOTALL)

    def expand_match(contraction):
        # Getting entire matched sub-string
        match = contraction.group(0)
        expanded_contraction = contraction_map.get(match)
        if not expand_contractions:
            print(match)
            return match
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# ### Data Extraction/Loading

# In[3]:


def load_data():
    filename1 = 'news_summary.csv'
    filename2 = 'news_summary_more.csv'
    df1 = pd.read_csv(filename1, encoding='iso-8859-1').reset_index(drop=True)
    df2 = pd.read_csv(filename2, encoding='iso-8859-1').reset_index(drop=True)
    
    df = data_preprocess(df1,df2)    
    return df


# ### Data Preprocessing

# In[4]:


def tokenizingText(texts,max_len=100): # breaking text into smaller words or subwords
    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(texts) # create vocabulary index based on word frequency if tokenize none
    seq = tokenizer.texts_to_sequences(texts)
    padded_seq = pad_sequences(seq,maxlen=max_len,padding='post')
    return padded_seq,tokenizer


# In[5]:


df = load_data()
ctext = df['text']
headlines = df['headlines']

maxlen = 100
padded_input_seq, in_tokenizer = tokenizingText(ctext, max_len=maxlen)
padded_target_seq, tar_tokenizer = tokenizingText(headlines, max_len=maxlen)


# In[6]:


# Using teacher forcing as a method for training RNN ( encode/decoder)
decode_padded_input_seq = padded_target_seq[:, :-1]
decode_padded_target_seq = padded_target_seq[:, 1:]


# In[7]:


print(len(in_tokenizer.word_index))
print(len(tar_tokenizer.word_index))


# ### Building a Model

# In[8]:


from tensorflow.keras import layers

class AttentionLayer(layers.Layer):
    def __init__(self):
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs):
        encoder_out, decoder_out = inputs
        attention = tf.reduce_sum(encoder_out * decoder_out, axis=-1)
        return attention

def apply_attention(inputs):
    encoder_outputs, decoder_outputs = inputs
    attention = Attention()([decoder_outputs, encoder_outputs])
    context_vector = tf.matmul(attention, encoder_outputs) 
    return context_vector


"""
Defining the Model Architecture
Builds a Seq2Seq model using work embeddings

Parameters: 
in_vocab_size : size of input vocabs
out_vocab_size: size of output vocabs
emb_dimension : Dimentsion of the word embeddings
units: no of units in RNN model

Implements:

-> Encoder :  
-> Decoder :
-> Attention :
-> Output layer : 
-> Model

output: keras model for training
"""

def create_model(in_vocab_size, out_vocab_size, emb_dimension, units):
    
    """Encoder"""
    encoder_in = Input(shape=(None,))
    encoder_emb = Embedding(in_vocab_size,emb_dimension)(encoder_in)
    encoder_lstm = Bidirectional(LSTM(units, return_sequences=True, return_state=True))
    encoder_out, forward_hidden, forward_cell, backward_hidden, backward_cell = encoder_lstm(encoder_emb)

    state_hidden = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([forward_hidden, backward_hidden])
    state_cell = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([forward_cell, backward_cell])

    """Attention"""
    attention = AttentionLayer()([encoder_out, encoder_out])
    
    
    """Decoder"""
    decoder_in = Input(shape=(None,))
    decoder_emb = Embedding(out_vocab_size, emb_dimension)(decoder_in)
    decoder_lstm = LSTM(units * 2, return_sequences=True, return_state=True)
    decoder_out, _, _ = decoder_lstm(decoder_emb, initial_state=[state_hidden, state_cell])

    """Output Layer"""
    decoder_dense_layer = TimeDistributed(Dense(out_vocab_size, activation='softmax'))
    outputs = decoder_dense_layer(decoder_out)

    """Building a model"""
    model = Model([encoder_in, decoder_in], outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

emb_dimension = 256
units = 128
in_vocab_size = len(in_tokenizer.word_index) + 1
out_vocab_size = len(tar_tokenizer.word_index) + 1

model = create_model(in_vocab_size, out_vocab_size, emb_dimension,units)
model.fit(
    [padded_input_seq, decode_padded_input_seq],
    np.expand_dims(decode_padded_target_seq, -1),
    batch_size=64,
    epochs=10,
    validation_split=0.2
)