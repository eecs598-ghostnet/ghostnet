import os, re, math, pickle
from io import open
import numpy as np
from english2phoneme import e2p

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.word2phonemes = {}
        self.phoneme2idx = {}
        self.idx2phoneme = []

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()

        # Tokenize lyrics 
        lyric_path = os.path.join(path, 'lyrics.txt')
        self.lyrics = self.tokenize(lyric_path)

        #Get or create phoneme dictionary if you recreate dictionary, you must recreate phonemes
        dict_path = os.path.join(path, 'w2p.pickle')
        phoneme_path = os.path.join(path, 'phonemes.txt')
        if os.path.exists(dict_path):
            self.dictionary.idx2phoneme, self.dictionary.phoneme2idx, self.dictionary.word2phonemes = pickle.load(open(dict_path, 'rb'))
            if os.path.exists(phoneme_path):
                self.phonemes = self.tokenize_phonemes(phoneme_path)
            else:
                self.phonemes = self.create_phonemes(phoneme_path)
        else:
            self.create_pdicts(dict_path)
            self.phonemes = self.create_phonemes(phoneme_path)       
            
    def tokenize(self, lyric_path, is_phonemes=False):
        """Tokenizes a text file."""
        assert os.path.exists(lyric_path)

        # Add words to the dictionary
        f_lyric = re.sub(' {2,}', ' ', re.sub('[\n]',' \n ', open(lyric_path, 'r', encoding="utf8").read().strip())).split(' ')
        self.dictionary.idx2word = list(set(f_lyric))
        self.dictionary.word2idx = {word:idx for idx,word in enumerate(self.dictionary.idx2word)}
        tokens = np.array([self.dictionary.word2idx[word] for word in f_lyric])

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
        

    def create_pdicts(self, dict_path):
        lyrics = ('<break>').join(self.dictionary.idx2word).lower()
        num_chunks = math.ceil(len(lyrics) / 100000)
        num_words = len(self.dictionary.idx2word)
        
        output = ''
        for i in range(num_chunks):
            output += e2p(('<break>').join(self.dictionary.idx2word[i*num_words//num_chunks:min((i+1)*num_words//num_chunks, num_words)]).lower())
        
        #Remove espeak extra symbols
        output = re.sub("[',%=_|;~]",'',output)
        output = re.sub("_:", '', output)

        # A quirk of this is that the phoneme for the word \n is ''
        output = re.sub(' ', '-', output).strip('\n').split('\n')
        output[output.index('')] = '\n'

        self.dictionary.idx2phoneme = list(set((('-').join(output).split('-'))))
        self.dictionary.phoneme2idx = {phoneme:idx for idx,phoneme in enumerate(self.dictionary.idx2phoneme)}

        w2p = [word.split('-') for word in output]
        for widx, w in enumerate(w2p): 
            for pidx, p in enumerate(w): 
                w2p[widx][pidx] = self.dictionary.phoneme2idx[p]
        
        self.dictionary.word2phonemes = w2p

        pickle.dump((self.dictionary.idx2phoneme, self.dictionary.phoneme2idx, self.dictionary.word2phonemes), open(dict_path,'wb'))

        
    def create_phonemes(self, phoneme_path):
        phonemes=[]

        for l in self.lyrics: 
            phonemes.append(self.dictionary.word2phonemes[l]) 
            
        
        output = ''
        for widx, w in enumerate(phonemes):
            if widx != 0:
                output += ' '
            for idx, p in enumerate(w):
                if idx != 0:
                    output+= '-'
                output+=self.dictionary.idx2phoneme[p]
            
        #Write to file
        output_file = open(phoneme_path, 'w+', encoding='utf8')
        output_file.write(output)

        return phonemes
