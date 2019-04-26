from __future__ import print_function
import random
import re

import colorama
import numpy as np

colorama.init()

def green_color(*args, **kwargs):
    print(colorama.Fore.GREEN, end='')
    print(*args, **kwargs)
    print(colorama.Style.RESET_ALL, end='')


def cyan_color(*args, **kwargs):
    print(colorama.Fore.CYAN, end='')
    print(*args, **kwargs)
    print(colorama.Style.RESET_ALL, end='')

def red_color(*args, **kwargs):
    print(colorama.Fore.RED, end='')
    print(*args, **kwargs)
    print(colorama.Style.RESET_ALL, end='')

def prediction_sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype(np.float64)
    preds += np.finfo(np.float64).tiny
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def tokenize_word_level(text):
    regexes = [
        (re.compile(r'(\s)"'), r'\1 “ '),
        (re.compile(r'([ (\[{<])"'), r'\1 “ '),
        (re.compile(r'([:,])([^\d])'), r' \1 \2'),
        (re.compile(r'([:,])$'), r' \1 '),
        (re.compile(r'\.\.\.'), r' ... '),
        (re.compile(r'([;@#$%&])'), r' \1 '),
        (re.compile(r'([?!\.])'), r' \1 '),
        (re.compile(r"([^'])' "), r"\1 ' "),
        (re.compile(r'([\]\[\(\)\{\}\<\>])'), r' \1 '),
        (re.compile(r'--'), r' -- '),
        (re.compile(r'"'), r' ” '),
        (re.compile(r"([^' ])('s|'m|'d) "), r"\1 \2 "),
        (re.compile(r"([^' ])('ll|'re|'ve|n't) "), r"\1 \2 "),
        (re.compile(r"\b(can)(not)\b"), r' \1 \2 '),
        (re.compile(r"\b(d)('ye)\b"), r' \1 \2 '),
        (re.compile(r"\b(gim)(me)\b"), r' \1 \2 '),
        (re.compile(r"\b(gon)(na)\b"), r' \1 \2 '),
        (re.compile(r"\b(got)(ta)\b"), r' \1 \2 '),
        (re.compile(r"\b(lem)(me)\b"), r' \1 \2 '),
        (re.compile(r"\b(mor)('n)\b"), r' \1 \2 '),
        (re.compile(r"\b(wan)(na)\b"), r' \1 \2 '),
        (re.compile(r'\n'), r' \\n ')
    ]

    text = " " + text + " "
    for regexp, substitution in regexes:
        text = regexp.sub(substitution, text)
    return text.split()

def detokenize_word_level(tokens):
    regexes = [
        (re.compile(r'[ ]?\\n[ ]?'), r'\n'),
        (re.compile(r"\b(can)\s(not)\b"), r'\1\2'),
        (re.compile(r"\b(d)\s('ye)\b"), r'\1\2'),
        (re.compile(r"\b(gim)\s(me)\b"), r'\1\2'),
        (re.compile(r"\b(gon)\s(na)\b"), r'\1\2'),
        (re.compile(r"\b(got)\s(ta)\b"), r'\1\2'),
        (re.compile(r"\b(lem)\s(me)\b"), r'\1\2'),
        (re.compile(r"\b(mor)\s('n)\b"), r'\1\2'),
        (re.compile(r"\b(wan)\s(na)\b"), r'\1\2'),
        (re.compile(r"([^' ]) ('ll|'re|'ve|n't)\b"), r"\1\2"),
        (re.compile(r"([^' ]) ('s|'m|'d)\b"), r"\1\2"),
        (re.compile(r'[ ]?”'), r'"'),
        (re.compile(r'[ ]?--[ ]?'), r'--'),
        (re.compile(r'([\[\(\{\<]) '), r'\1'),
        (re.compile(r' ([\]\)\}\>])'), r'\1'),
        (re.compile(r'([\]\)\}\>]) ([:;,.])'), r'\1\2'),
        (re.compile(r"([^']) ' "), r"\1' "),
        (re.compile(r' ([?!\.])'), r'\1'),
        (re.compile(r'([^\.])\s(\.)([\]\)}>"\']*)\s*$'), r'\1\2\3'),
        (re.compile(r'([#$]) '), r'\1'),
        (re.compile(r' ([;%:,])'), r'\1'),
        (re.compile(r'(“)[ ]?'), r'"')
    ]

    text = ' '.join(tokens)
    for regexp, substitution in regexes:
        text = regexp.sub(substitution, text)
    return text.strip()

def seed_search(text, num_seeds=50, max_seed_length=50):
    lines = text.split('\n')
    if len(lines) > num_seeds * 4:
        lines = random.sample(lines, num_seeds * 4)
    lines = sorted(lines, key=len, reverse=True)
    lines = lines[:num_seeds]
    return [line[:max_seed_length].rsplit(None, 1)[0] for line in lines]

def stateful_RNN_shape(data, batch_size, seq_length, seq_step):
    inputs = data[:-1]
    targets = data[1:]

    inputs = _sequence_creator(inputs, seq_length, seq_step)
    targets = _sequence_creator(targets, seq_length, seq_step)

    inputs = _stateful_rnn_batch_sort(inputs, batch_size)
    targets = _stateful_rnn_batch_sort(targets, batch_size)

    targets = targets[:, :, np.newaxis]
    return inputs, targets

def _sequence_creator(vector, seq_length, seq_step):
    passes = []
    for offset in range(0, seq_length, seq_step):
        pass_samples = vector[offset:]
        number_pass_samples = pass_samples.size // seq_length
        pass_samples = np.resize(pass_samples,
                                 (number_pass_samples, seq_length))
        passes.append(pass_samples)
    return np.concatenate(passes)

def _stateful_rnn_batch_sort(sequences, batch_size):
    num_batches = sequences.shape[0] // batch_size
    num_samples = num_batches * batch_size
    reshuffled_samples = np.zeros((num_samples, sequences.shape[1]), dtype=np.int32)
    for batch_index in range(batch_size):
        slice_start = batch_index * num_batches
        slice_end = slice_start + num_batches
        index_slice = sequences[slice_start:slice_end, :]
        reshuffled_samples[batch_index::batch_size, :] = index_slice
    return reshuffled_samples
