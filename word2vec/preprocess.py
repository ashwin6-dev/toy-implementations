import re
import numpy as np
import torch

def get_words(text):
    return re.findall(r"\b\w+\b", text.lower())

def word_to_id(vocab):
    word_ids = {}
    vocab_size = 0

    for i, word in enumerate(vocab):
        if word in word_ids:
            continue

        word_ids[word] = vocab_size
        vocab_size += 1

    return word_ids, vocab_size

def one_hot(indices, vocab_size):
    return np.eye(vocab_size, dtype=np.float32)[indices]

def build_dataset(text):
    words = get_words(text)
    word_ids, vocab_size = word_to_id(words)

    inputs, outputs = [], []

    for i, word in enumerate(words):
        adj_words = []

        if i > 0:
            adj_words.append(words[i - 1])

        if i < len(words) - 1:
            adj_words.append(words[i + 1])

        for adj_word in adj_words:
            inputs.append(word_ids[word])
            outputs.append(word_ids[adj_word]) 

    inputs = torch.from_numpy(one_hot(inputs, vocab_size))
    outputs = torch.from_numpy(one_hot(outputs, vocab_size))

    return inputs, outputs, word_ids, vocab_size