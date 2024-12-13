# RNN-Based Abstractive Text Summarization

This repository contains a Python implementation of a Recurrent Neural Network (RNN) for abstractive text summarization. The model leverages deep learning techniques to generate concise and meaningful summaries from text input, evaluated using ROUGE and BLEU metrics.

## Features

- Abstractive text summarization using RNN architecture.
- Automatic evaluation with ROUGE and BLEU scores.
- Tokenization and preprocessing for input text data.

## Prerequisites

- Python 3.7 or higher
- TensorFlow 2.x
- NumPy
- NLTK
- Rouge
- Scikit-learn

### Installation

1. Clone this repository:
    git clone https://github.com/skazim/textProcessing_AbsSum/
    cd rnn-text-summarization

2. Install the required Python packages:
    pip install -r requirements.txt

## Evaluation Metrics
1. ROUGE: Measures the overlap between predicted and reference n-grams.
2. BLEU: Evaluates the precision of n-grams in predicted summaries against the references.

## Limitations
1. Struggles with long-range dependencies in lengthy texts.
2. Performance depends heavily on the quality of the training dataset.
3. Computationally expensive, especially for large datasets and long sequences.


## Acknowledgments
The ROUGE and BLEU implementations used in this project are based on standard libraries.
Thanks to the deep learning community for providing resources and tutorials.