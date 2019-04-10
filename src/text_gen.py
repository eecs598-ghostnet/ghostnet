import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import re
import sys

from text_generation_model import TextGenerationModel
from dataloader import get_dataloader
import utils
import config

def gen_text(model, txt_vocab, phoneme_vocab, corpus, device, seed_words='\n', max_length=50, prevent_double_newlines=True):
    # TODO seed with gaussian distr of LSTM hidden state vector
    d = corpus.dictionary
    d.word2phonemes['<sos>'] = ['<sos>', '<eos>']

    # Seed LSTM with given phrase
    words = re.findall(r'\S+|\n', seed_words)
    hidden = None
    for word in words[:-1]:
        word_token = txt_vocab.stoi[word]
        txt_input = torch.Tensor([word_token]).view(1,1).to(device).long()

        phonemes = d.word2phonemes[word]
        phoneme_tokens = [phoneme_vocab.stoi[phoneme] for phoneme in phonemes]
        phonemes_input = torch.Tensor(phoneme_tokens).view(1,1,-1).to(device).long()

        #outputs = model(txt_input, phonemes_input, txt_lengths, phoneme_lengths)
        outputs, hidden = model.gen_word(txt_input, phonemes_input, hidden=hidden)


    word = words[-1]

    for i in range(max_length):
        word_token = txt_vocab.stoi[word]
        txt_input = torch.Tensor([word_token]).view(1,1).to(device).long()

        phonemes = d.word2phonemes[word]
        phoneme_tokens = [phoneme_vocab.stoi[phoneme] for phoneme in phonemes]
        phonemes_input = torch.Tensor(phoneme_tokens).view(1,1,-1).to(device).long()

        #outputs = model(txt_input, phonemes_input, txt_lengths, phoneme_lengths)
        outputs, hidden = model.gen_word(txt_input, phonemes_input, hidden=hidden)

        top_outs = torch.topk(outputs, k=3)

        # Get top non-unk word
        score, token = None, None
        for i in range(top_outs[1].size(0)):
            score, token = top_outs[0][i], top_outs[1][i]
            if token == txt_vocab.stoi['<unk>']:
                print('skipping unk')
                continue
            if prevent_double_newlines and word == '\n' and token == txt_vocab.stoi['\n']:
                print('skipping double newline')
                continue

            assert token != txt_vocab.stoi['<unk>']
            assert txt_vocab.itos[token] != '<unk>'
            break

        #print(score, token)
        word = txt_vocab.itos[token]
        assert word != '<unk>', "how is this happening"

        if word == '<eos>':
            print('<eos> found')
            break

        words.append(word)

    print()
    print(' '.join(words))


def load_model(state_dict_path, device, **kwargs):
    model = TextGenerationModel(**kwargs).to(device)
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':
    # vocab must be shared with trained modeljkk
    artist_dir = '../data/lyrics/combined_trunc'
    _, txt_vocab, phoneme_vocab, corpus = get_dataloader(artist_dir, min_vocab_freq=3)

    seed_words = '<sos>'
    if len(sys.argv) > 1:
        seed_words = sys.argv[1]

    # TODO shared config for these
    model_params = config.get_model_params(txt_vocab, phoneme_vocab)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('Loading model weights...')
    model = load_model('../model/pretrain_ae_55.pt', device, **model_params)
    print('Done')

    gen_text(model, txt_vocab, phoneme_vocab, corpus, device, seed_words=seed_words, max_length=50)
