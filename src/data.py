import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word2phonemes = {}
        self.phoneme2idx = {}
        self.idx2phoneme = []

    def add_word(self,word, phonemes):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word)-1
            self.word2phonemes[word] = []
            for phoneme in phonemes.strip('-').split('-'):
                if phoneme not in self.phoneme2idx:
                    self.idx2phoneme.append(phoneme)
                    self.phoneme2idx[phoneme] = len(self.idx2phoneme)-1
                self.word2phonemes[word].append(self.phoneme2idx[phoneme])
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.lyrics, self.phonemes = self.tokenize(os.path.join(path, 'lyrics.txt'), os.path.join(path,'phonemes.txt'))

    def tokenize(self, lyric_path, phoneme_path):
        """Tokenizes a text file."""
        assert os.path.exists(lyric_path)
        assert os.path.exists(phoneme_path)
        # Add words to the dictionary
        f_lyric = open(lyric_path, 'r', encoding="utf8").read().strip().split('\n')
        f_phoneme = open(phoneme_path, 'r', encoding='utf8').read().strip().split('\n')
        
        tokens = 0
        for lidx, line in enumerate(f_lyric):
            phoneme_line = f_phoneme[lidx]
            words = line.split() + ['<eos>']
            phonemes = phoneme_line.split() + ['<eos>']
            tokens += len(words)
            for widx,word in enumerate(words):
                if len(words) != len(phonemes):
                    import IPython; IPython.embed()
                phoneme = phonemes[widx]
                self.dictionary.add_word(word, phoneme)

        # Tokenize file content
        ids = []
        phonemes = []
        token = 0
        for lidx, line in enumerate(f_lyric):
            words = line.split() + ['<eos>']
            for word in words:
                ids.append(self.dictionary.word2idx[word])
                phonemes.append(self.dictionary.word2phonemes[word])
                token += 1

        return ids, phonemes

