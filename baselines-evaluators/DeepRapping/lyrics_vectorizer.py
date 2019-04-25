from __future__ import print_function
from collections import Counter
import numpy as np
from utilities import red_color
from utilities import tokenize_word_level, detokenize_word_level

class Lyrics_Vectorizer:
    def __init__(self, text, word_level_flag, pristine_input, pristine_output):
        self.word_level_flag = word_level_flag
        self._pristine_input = pristine_input
        self._pristine_output = pristine_output

        tokens = self._tokenizer(text)
        print('Data length:', len(tokens))
        token_counts = Counter(tokens)
        tokens = [x[0] for x in token_counts.most_common()]
        self._token_indices = {x: i for i, x in enumerate(tokens)}
        self._indices_token = {i: x for i, x in enumerate(tokens)}
        self.vocab_size = len(tokens)
        print('Vocabulary Size:', self.vocab_size)

    def _tokenizer(self, text):
        if not self._pristine_input:
            text = text.lower()
        if self.word_level_flag:
            if self._pristine_input:
                return text.split()
            return tokenize_word_level(text)
        return text

    def _detokenizer(self, tokens):
        if self.word_level_flag:
            if self._pristine_output:
                return ' '.join(tokens)
            return detokenize_word_level(tokens)
        return ''.join(tokens)

    def vectorize(self, text):
        tokens = self._tokenizer(text)
        indices = []
        for token in tokens:
            if token in self._token_indices:
                indices.append(self._token_indices[token])
            else:
                red_color('Ignored Unfamiliar Token:', token)
        return np.array(indices, dtype=np.int32)

    def unvectorize(self, vector):
        tokens = [self._indices_token[index] for index in vector.tolist()]
        return self._detokenizer(tokens)
