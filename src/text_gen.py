import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from text_generation_model import TextGenerationModel
from dataloader import get_dataloader
import utils

def gen_text(model, txt_vocab, phoneme_vocab, corpus, device, seed_word='\n', max_length=50):
    # TODO seed with gaussian distr of LSTM hidden state vector
    d = corpus.dictionary

    word, hidden = seed_word, None

    words = [word]
    for i in range(max_length):
        word_token = txt_vocab.stoi[word]
        txt_input = torch.Tensor([word_token]).view(1,1).to(device).long()

        phonemes = d.word2phonemes[word]
        phoneme_tokens = [phoneme_vocab.stoi[phoneme] for phoneme in phonemes]
        phonemes_input = torch.Tensor(phoneme_tokens).view(1,1,-1).to(device).long()

        #outputs = model(txt_input, phonemes_input, txt_lengths, phoneme_lengths)
        outputs, hidden = model.gen_word(txt_input, phonemes_input, hidden=hidden)

        top_outs = torch.topk(outputs, k=2)

        # Get top non-unk word
        score, token = None, None
        if top_outs[1][0] != phoneme_vocab.stoi['<unk>']:
            score, token = top_outs[0][0], top_outs[1][0]
        else:
            score, token = top_outs[0][1], top_outs[1][1]

        #print(score, token)
        word = txt_vocab.itos[token]

        if word == '<pad>':
            break

        words.append(word)

    print(' '.join(words))


def load_model(state_dict_path, device, **kwargs):
    model = TextGenerationModel(**kwargs)

    try:
        state_dict = torch.load(state_dict_path)
    except RuntimeError:
        state_dict = torch.load(state_dict_path, map_location='cpu')
    model.load_state_dict(state_dict)

    return model.to(device)


if __name__ == '__main__':
    # vocab must be shared with trained model
    artist_dir = '../data/lyrics/combined_trunc'
    _, txt_vocab, phoneme_vocab, corpus = get_dataloader(artist_dir, min_vocab_freq=3)

    # TODO shared config for these
    model_params = {
        'vocab_size': len(txt_vocab),
        'embed_size': 50,
        'hidden_size': 100,
        'phoneme_vocab_size': len(phoneme_vocab),
        'phoneme_embed_size': 30,
        'phoneme_hidden_size': 40,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model('../model/2unk_no_chorus_trunc.pt', device, **model_params)

    gen_text(model, txt_vocab, phoneme_vocab, corpus, device, seed_word='you')
