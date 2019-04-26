import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import copy
import random
import re
import sys
from tqdm import tqdm

from text_generation_model import TextGenerationModel, AttentionEncoderDecoder
from dataloader import get_dataloader
import utils
import config


def get_seed_state(model, seed_words, d, txt_vocab, phoneme_vocab, device):
    scores, hidden = None, None
    for word in seed_words:
        word_token = txt_vocab.stoi[word]
        txt_input = torch.Tensor([word_token]).view(1,1).to(device).long()

        phonemes = d.word2phonemes[word]
        phoneme_tokens = [phoneme_vocab.stoi[phoneme] for phoneme in phonemes]
        phonemes_input = torch.Tensor(phoneme_tokens).view(1,1,-1).to(device).long()

        #outputs = model(txt_input, phonemes_input, txt_lengths, phoneme_lengths)
        scores, hidden = model.gen_word(txt_input, phonemes_input, hidden=hidden)

    return scores, hidden


# At each next step, consider next scores as model(words[-1], hidden)
class BeamPath():
    def __init__(self, words, hidden, sum_logodds=0, finished=False):
        self.words = words
        self.hidden = hidden
        self.sum_logodds = sum_logodds
        self.finished = finished

    def value(self):
        return self.sum_logodds / len(self.words)
        return self.sum_logodds

    def get_next_paths(self, hidden, scores, txt_vocab, k=None):
        next_paths = []

        if k is None:
            for token, score in enumerate(scores):
                if txt_vocab.itos[token] != '<unk>':
                    next_paths.append(BeamPath(
                        self.words + [txt_vocab.itos[token]],
                        hidden,
                        sum_logodds=self.sum_logodds + score,
                        finished=(txt_vocab.itos[token]=='<eos>')
                    ))
        else:
            top_outs = torch.topk(scores, k=k+1)
            next_paths = [BeamPath(
                self.words + [txt_vocab.itos[token]],
                hidden,
                sum_logodds=self.sum_logodds + score,
                finished=(txt_vocab.itos[token]=='<eos>')
            ) for score, token in zip(*top_outs)
            if txt_vocab.itos[token] != '<unk>']
        return next_paths

    def __repr__(self):
        return """Logodds value: {}\n{}""".format(
            self.value().item(), ' '.join(self.words))


def beam_search(model, txt_vocab, phoneme_vocab, corpus, device, k=3,
                seed_words='<sos>', max_length=50):
    d = corpus.dictionary
    d.word2phonemes['<sos>'] = ['<sos>', '<eos>']

    # Seed LSTM with given phrase
    words = re.findall(r'\S+|\n', seed_words)

    scores, hidden = get_seed_state(model, words, d, txt_vocab,
                                    phoneme_vocab, device)

    # Start by considering top k from seed state
    top_outs = torch.topk(scores, k=k+1)
    paths = [BeamPath(words + [txt_vocab.itos[token]], hidden, score.item())
             for score, token in zip(*top_outs)
             if txt_vocab.itos[token] != '<unk>' and txt_vocab.itos[token] != '<eos>']

    for i in range(max_length):
        if not sum([not path.finished for path in paths]):
            print('All paths finished')
            break

        next_paths = []

        # Look at next scores for each path
        for path in paths:
            if path.finished:
                next_paths.append(path)
            else:
                word_token = txt_vocab.stoi[path.words[-1]]
                txt_input = torch.Tensor([word_token]).view(1,1).to(device).long()

                phonemes = d.word2phonemes[path.words[-1]]
                phoneme_tokens = [phoneme_vocab.stoi[phoneme] for phoneme in phonemes]
                phonemes_input = torch.Tensor(phoneme_tokens).view(1,1,-1).to(device).long()

                scores, hidden = model.gen_word(txt_input, phonemes_input, hidden=path.hidden)
                #top_outs = torch.topk(scores, )

                next_paths.extend(path.get_next_paths(hidden, scores, txt_vocab, k=2*k))

        # Get top k paths
        next_paths.sort(key=lambda path:path.value())
        paths = next_paths[-k:]

    #print('Top k beam search paths:')
    #for i, path in enumerate(paths):
    #    print('\nPATH {}'.format(k-i))
    #    print(path)

    return ' '.join(paths[-1].words)


class AttnBeamPath(BeamPath):
    def __init__(self, words, encodings, enc_hidden, dec_hidden, sum_logodds=0, finished=False):
        self.words = words
        self.encodings = encodings
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden
        self.sum_logodds = sum_logodds
        self.finished = finished

    def value(self):
        return self.sum_logodds / len(self.words)

    def __repr__(self):
        return """Logodds: {}\n{}""".format(
            self.value(), ' '.join(self.words))

    def get_next_paths(self, scores, encodings, enc_hidden, dec_hidden, txt_vocab, k):
        next_paths = []

        top_outs = torch.topk(scores, k=k+1)
        next_paths = [AttnBeamPath(
            self.words + [txt_vocab.itos[token]],
            encodings, enc_hidden, dec_hidden,
            sum_logodds=self.sum_logodds+score,
            finished=(txt_vocab.itos[token]=='<eos>')
        ) for score, token in zip(*top_outs)
        if txt_vocab.itos[token] != '<unk>']

        return next_paths


def attn_beam_search(model, txt_vocab, phoneme_vocab, corpus, device, k=3,
                     seed_words='<sos>', max_length=50, show_options=False):
    d = corpus.dictionary
    d.word2phonemes['<sos>'] = ['<sos>', '<eos>']

    # Seed LSTM with given phrase
    words = re.findall(r'\S+|\n', seed_words)

    # Get seed state of model
    scores, encodings, enc_hidden, dec_hidden = None, [], None, None
    for word in words:
        word_token = txt_vocab.stoi[word]
        txt_input = torch.Tensor([word_token]).view(1,1).to(device).long()

        phonemes = d.word2phonemes[word]
        phoneme_tokens = [phoneme_vocab.stoi[phoneme] for phoneme in phonemes]
        phonemes_input = torch.Tensor(phoneme_tokens).view(1,1,-1).to(device).long()

        #outputs = model(txt_input, phonemes_input, txt_lengths, phoneme_lengths)
        scores, encodings, enc_hidden, dec_hidden = model.gen_word(
            txt_input, phonemes_input, encodings, enc_hidden, dec_hidden
        )

    # Consider top k from seed state
    top_outs = torch.topk(scores, k=k+1)
    paths = [AttnBeamPath(words + [txt_vocab.itos[token]], encodings,
                          enc_hidden, dec_hidden, score.item())
             for score, token in zip(*top_outs)
             if txt_vocab.itos[token] != '<unk>' and txt_vocab.itos[token] != '<eos>']

    for i in range(max_length):
        if not sum([not path.finished for path in paths]):
            if show_options:
                print('All paths finished')
            break

        # Find top k scores for each path
        next_paths = []
        for path in paths:
            if path.finished:
                next_paths.append(path)
            else:
                encodings = copy.copy(path.encodings)
                word_token = txt_vocab.stoi[path.words[-1]]
                txt_input = torch.Tensor([word_token]).view(1,1).to(device).long()
                phonemes = d.word2phonemes[path.words[-1]]
                phoneme_tokens = [phoneme_vocab.stoi[phoneme] for phoneme in phonemes]
                phonemes_input = torch.Tensor(phoneme_tokens).view(1,1,-1).to(device).long()

                try:
                    scores, encodings, enc_hidden, dec_hidden = model.gen_word(
                        txt_input, phonemes_input, encodings, path.enc_hidden, path.dec_hidden
                    )
                except IndexError as e:
                    print(f'IndexError: {e}')
                    continue

                next_paths.extend(path.get_next_paths(
                    scores, encodings, enc_hidden, dec_hidden, txt_vocab, k=2*k
                ))

            # Get top k paths
            next_paths.sort(key=lambda path:path.value())
            paths = next_paths[-k:]

    if show_options:
        print('Top k beam search paths:')
        for i, path in enumerate(paths):
            print('\nPATH {}'.format(k-i))
            print(path)

    return ' '.join(paths[-1].words)




def greedy_search(model, txt_vocab, phoneme_vocab, corpus, device,
                  seed_words='\n', max_length=50,
                  prevent_double_newlines=True):
    # TODO seed with gaussian distr of LSTM hidden state vector
    d = corpus.dictionary
    d.word2phonemes['<sos>'] = ['<sos>', '<eos>']

    # Seed LSTM with given phrase
    words = re.findall(r'\S+|\n', seed_words)

    scores, hidden = get_seed_state(model, words, d, txt_vocab,
                                    phoneme_vocab, device)

    for i in range(max_length):
        top_outs = torch.topk(scores, k=3)

        # Get top non-unk word
        score, token = None, None
        for i in range(top_outs[1].size(0)):
            score, token = top_outs[0][i], top_outs[1][i]
            if token == txt_vocab.stoi['<unk>']:
                print('skipping unk')
                continue
            if prevent_double_newlines and words[-1] == '\n' and token == txt_vocab.stoi['\n']:
                print('skipping double newline')
                continue

            assert token != txt_vocab.stoi['<unk>']
            assert txt_vocab.itos[token] != '<unk>'
            break

        #print(score, token)
        word = txt_vocab.itos[token]
        if word == '<eos>':
            print('<eos> found')
            break

        words.append(word)

        word_token = txt_vocab.stoi[word]
        txt_input = torch.Tensor([word_token]).view(1,1).to(device).long()

        phonemes = d.word2phonemes[word]
        phoneme_tokens = [phoneme_vocab.stoi[phoneme] for phoneme in phonemes]
        phonemes_input = torch.Tensor(phoneme_tokens).view(1,1,-1).to(device).long()


        scores, hidden = model.gen_word(txt_input, phonemes_input, hidden=hidden)


    print()
    print(' '.join(words))


def gen_samples(seed_phrases, *args, **kwargs):
    for seed_words in seed_phrases:
        words = attn_beam_search(*args, **kwargs, seed_words=seed_words)
        print('-------------------------')
        print('Seed: "{}"'.format(seed_words))
        print('Generated:')
        print(words)
        print()

def gen_random_samples(num_samples, txt_vocab, *args, **kwargs):
    for i in range(num_samples):
        word = random.choice(txt_vocab.itos)
        seed_words = '<sos> ' + word
        words = attn_beam_search(*args, **kwargs, seed_words=seed_words)
        print()
        print(words)



if __name__ == '__main__':
    # vocab must be shared with trained modeljkk
    artist_dir = '../data/lyrics/combined_trunc'
    model_weights_path = '../model/attention/final.pt'
    ModelType = AttentionEncoderDecoder

    if len(sys.argv) > 1:
        seed_words = '<sos> ' + sys.argv[1]
    else:
        seed_words = None

    # Get dataloaders
    dataloaders, txt_vocab, phoneme_vocab, corpus = get_dataloader(
        artist_dir, batch_sizes=(12, 12, 12), min_vocab_freq=1, max_len=256
    )
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
    print(f'train_dataset size: {dataset_sizes["train"]}')

    # Max length needed for non-positional attention
    max_length = max([len(example.text) for x in ['train', 'val', 'test']
                      for example in dataloaders[x].dataset.examples])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get model
    model_params = config.get_model_params(txt_vocab, phoneme_vocab)
    model_params['max_length'] = max_length

    model = config.load_model(ModelType, model_weights_path, device, **model_params)


    if seed_words is not None:
        #greedy_search(model, txt_vocab, phoneme_vocab, corpus, device, seed_words=seed_words, max_length=50)
        attn_beam_search(model, txt_vocab, phoneme_vocab, corpus, device, seed_words=seed_words, max_length=50, k=5, show_options=True)

    else:
        seed_phrases = [
            '<sos>',
            '<sos> we got',
            '<sos> test this',
            '<sos> all money aint good money',
            '<sos> what is it',
            '<sos> ugh yea',
            '<sos> words embedded',
            '<sos> i love the',
            '<sos> yo yo',
        ]
        #gen_samples(seed_phrases, model, txt_vocab, phoneme_vocab, corpus, device, max_length=40, k=5)
        gen_random_samples(1000, txt_vocab, model, txt_vocab, phoneme_vocab, corpus, device, max_length=40, k=5)
