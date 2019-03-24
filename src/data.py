import os, re, math, pickle
from io import open
import numpy as np
from english2phoneme import e2p

class Dictionary(object):
    def __init__(self, dict_path=None):
        if dict_path:
            self.word2phonemes, self.idx2phoneme = pickle.load(open(dict_path, 'rb'))
            self.idx2word = self.word2phonemes.keys()
            self.word2idx = {word:idx for idx,word in enumerate(self.idx2word)}
            self.phoneme2idx = {phoneme:idx for idx,phoneme in enumerate(self.idx2phoneme)}

            #Convert dictionary to token form
            self.word2phonemes_idx = [[] for word in self.idx2word]
            for word in self.word2phonemes:
                self.word2phonemes_idx.append([])
                for phoneme in self.word2phonemes[word]:
                    self.word2phonemes_idx[self.word2idx[word]].append(self.phoneme2idx[phoneme])


        else:
            self.word2phonemes = {}
            self.word2phonemes_idx = []
            self.word2idx = {}
            self.idx2word = []
            self.phoneme2idx = {}
            self.idx2phoneme = []

class Corpus(object):
    def __init__(self, path):
        lyric_path = os.path.join(path, 'lyrics.txt')
        phoneme_path = os.path.join(path, 'phonemes.txt')
        dict_path = os.path.join(path, 'dict.pickle')

        if os.path.exists(dict_path) and os.path.exists(phoneme_path):
            self.dictionary = Dictionary(dict_path)
            lyrics_list = self.process_lyrics(lyric_path)
            self.lyrics = self.tokenize_text(lyrics_list)
            self.phonemes = self.tokenize_phonemes(phoneme_path)
        else:
            self.dictionary = Dictionary()
            self.create_dictionaries(lyric_path, phoneme_path, dict_path)

    def create_dictionaries(self, lyric_path, phoneme_path, dict_path):
        lyrics_list = self.process_lyrics(lyric_path)
        self.dictionary.idx2word = list(set(lyrics_list))
        self.dictionary.word2idx = {word:idx for idx,word in enumerate(self.dictionary.idx2word)}
        self.lyrics = self.tokenize_text(lyrics_list)
        self.create_pdicts()
        self.phonemes = self.create_phonemes(phoneme_path)
        pickle.dump((self.dictionary.word2phonemes, self.dictionary.idx2phoneme), open(dict_path, 'wb'))

    def process_lyrics(self, lyric_path):
        f_lyric = open(lyric_path, 'r', encoding='utf8').read().strip()
        lyrics_list = re.sub('[\n]',' \n ', f_lyric)
        lyrics_list = re.sub(' {2,}', ' ', lyrics_list)
        lyrics_list = lyrics_list.split(' ')
        return lyrics_list

    def tokenize_text(self, lyrics_list):
        """Tokenizes a text file."""
        tokens = np.array([self.dictionary.word2idx[word] for word in lyrics_list])
        return tokens

    def tokenize_phonemes(self, phoneme_path):
        assert os.path.exists(phoneme_path)
        phonemes = []

        f_phoneme = open(phoneme_path, 'r', encoding='utf8').read().split(' ')
        for w in f_phoneme:
            ps = w.split('-')
            pword = []
            for p in ps:
                pword.append(self.dictionary.phoneme2idx[p])
            phonemes.append(pword)
        return phonemes


    def create_pdicts(self):
        lyrics = ('<break>').join(self.dictionary.idx2word).lower()
        num_chunks = int(math.ceil(len(lyrics) / 100000.0))
        num_words = len(self.dictionary.idx2word)

        output = ''
        for i in range(num_chunks):
            start_idx = i*num_words//num_chunks
            stop_idx = min((i+1)*num_words//num_chunks, num_words)
            words = self.dictionary.idx2word[start_idx:stop_idx]
            output += e2p(('<break>').join(words))

        #Remove espeak extra symbols
        output = re.sub("[',%=_|;~]",'',output)
        output = re.sub("_:", '', output)

        # A quirk of the way we are using espeak is that the phoneme for the word \n is '', so we sub it with '\n'
        output = re.sub(' ', '-', output).strip('\n').split('\n')
        output[output.index('')] = '\n'

        word_list =  (('-').join(output)).split('-')
        self.dictionary.idx2phoneme = list(set(word_list))
        self.dictionary.phoneme2idx = {phoneme:idx for idx,phoneme in enumerate(self.dictionary.idx2phoneme)}

        w2p = {word:phonemes.split('-') for [word,phonemes] in zip(self.dictionary.idx2word, output)}
        w2p_i = [word.split('-') for word in output]
        for widx, w in enumerate(w2p_i):
            for pidx, p in enumerate(w):
                w2p_i[widx][pidx] = self.dictionary.phoneme2idx[p]

        self.dictionary.word2phonemes = w2p
        self.dictionary.word2phonemes_idx = w2p_i

    def create_phonemes(self, phoneme_path):
        phonemes=[]
        output = ''
        for lidx, l in enumerate(self.lyrics):
            phonemes.append(self.dictionary.word2phonemes_idx[l])
            if lidx != 0:
                output += ' '
            output+= ('-').join(self.dictionary.word2phonemes[self.dictionary.idx2word[l]])

        #Write to file
        output_file = open(phoneme_path, 'w+', encoding='utf8')
        output_file.write(output)

        return phonemes

    def build_vocab_to_corpus_idx_translate(self, vocab):
        d = self.dictionary
        self.idx_translate = {vidx:d.word2idx[word]
                              for word, vidx in vocab.stoi.items()
                              if word not in ('<unk>', '<pad>')}

        self.pad_token_vidx = vocab.stoi['<pad>']
        self.unk_token_vidx = vocab.stoi['<unk>']

        # TODO does it matter that can't index this to an actual phoneme?
        # TODO do the negative values work/matter for embedding/prediction?
        #self.pad_token_cidx = len(d.idx2word)   # set pad token value
        self.pad_token_cidx = -1                 # set pad token value
        self.unk_token_cidx = -2                 # do these values work/matter?


    def vocab_word_to_phoneme_idxs(self, vidx):
        """
        Translate a vocab word index into a the corresponding phoneme indices
        according to idx_translate. idx_translate maps vocab word indices to
        corpus word indices (because vocabs are built separately).
        """
        assert self.idx_translate is not None, "Must build translation first"
        translate = self.idx_translate

        if vidx == self.pad_token_vidx:
            return [self.pad_token_cidx]
        if vidx == self.unk_token_vidx:
            return [self.unk_token_cidx]

        d = self.dictionary
        return d.word2phonemes_idx[translate[vidx]]
        #return [d.idx2phoneme[pidx] for pidx in pidxs]
