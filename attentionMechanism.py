import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Embedding, Bidirectional, TimeDistributed, Attention
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data['ctext'], data['headlines']


def preprocess_texts(texts, tokenizer=None, max_len=100):
    if tokenizer is None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded, tokenizer

def build_model(input_vocab_size, output_vocab_size, embedding_dim, units):
    # Encoder
    encoder_inputs = Input(shape=(None,))
    enc_emb = Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm = Bidirectional(LSTM(units, return_sequences=True, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(enc_emb)
    state_h = tf.concat([forward_h, backward_h], axis=-1)
    state_c = tf.concat([forward_c, backward_c], axis=-1)

    # Attention
    attention = Attention()([encoder_outputs, encoder_outputs])

    # Decoder
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(output_vocab_size, embedding_dim)
    dec_emb = dec_emb_layer(decoder_inputs)
    decoder_lstm = LSTM(units * 2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
    attn_out = tf.concat([decoder_outputs, attention], axis=-1)

    # Output layer
    decoder_dense = TimeDistributed(Dense(output_vocab_size, activation='softmax'))
    outputs = decoder_dense(attn_out)

    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


file_path = 'dataset_summary.csv'
input_texts, target_texts = load_data(file_path)
print(input_texts, target_texts)

# max_seq_len = 100
# input_sequences, input_tokenizer = preprocess_texts(input_texts, max_len=max_seq_len)
# target_sequences, target_tokenizer = preprocess_texts(target_texts, max_len=max_seq_len)

# print(input_sequences)
# print()

# decoder_input_sequences = target_sequences[:, :-1]
# decoder_target_sequences = target_sequences[:, 1:]

# embedding_dim = 256
# units = 128
# input_vocab_size = len(input_tokenizer.word_index) + 1
# output_vocab_size = len(target_tokenizer.word_index) + 1

# model = build_model(input_vocab_size, output_vocab_size, embedding_dim, units)
# model.fit(
#     [input_sequences, decoder_input_sequences],
#     np.expand_dims(decoder_target_sequences, -1),
#     batch_size=64,
#     epochs=10,
#     validation_split=0.2
# )


