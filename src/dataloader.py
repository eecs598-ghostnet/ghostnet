import csv
import os
import re
from math import gcd
from torchtext.data import *
from torchtext.data.dataset import *

from tqdm import tqdm

import data


class StanzaDataset(Dataset):
    """
    Slightly altered Dataset from torchtext.data.TabularDataset that can read
    stanzas delimited by double newlines.
    """

    def __init__(self, path, format, fields, corpus, skip_header=False,
                 csv_reader_params={}, **kwargs):
        format = format.lower()
        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromCSV, 'csv': Example.fromCSV}[format]

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            reader = stanza_phoneme_reader(f, corpus)

            if format in ['csv', 'tsv'] and isinstance(fields, dict):
                if skip_header:
                    raise ValueError('When using a dict to specify fields with a {} file,'
                                     'skip_header must be False and'
                                     'the file must have a header.'.format(format))
                header = next(reader)
                field_to_index = {f: header.index(f) for f in fields.keys()}
                make_example = partial(make_example, field_to_index=field_to_index)

            if skip_header:
                next(reader)

            examples = [make_example(line, fields) for line in reader]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        super(StanzaDataset, self).__init__(examples, fields, **kwargs)


def stanza_reader(f):
    stanza = ''
    for line in f:
        if line == '\n':
            if stanza:
                yield [stanza]
                stanza = ''
        else:
            stanza += line


def stanza_phoneme_reader(f, corpus):
    d = corpus.dictionary

    stanza = ''
    for line in f:
        if line == '\n':
            if stanza:
                # TODO do we want these newline tokens to have a phoneme mapping?
                # TODO why is there an extra newline? does this match anything in the words?
                phonemes = [' '.join(d.word2phonemes[word]) for word in re.findall(r'\S+|\n', stanza)[:-1]]
                yield [stanza, phonemes]
                stanza = ''
        else:
            stanza += line


def song_reader(f):
    song = ''
    prev_line = None
    for line in f:
        if line == '\n' and prev_line == '\n':
            if song:
                yield song.strip() + '\n'
                song = ''
        else:
            song += line
            prev_line = line


def split_songs(artist_dir, train_pct=80, val_pct=15, test_pct=5):
    split_gcd = gcd(gcd(train_pct, val_pct), gcd(val_pct, test_pct))
    train_ct, val_ct, test_ct = train_pct//split_gcd, val_pct//split_gcd, test_pct//split_gcd

    lyrics_path = os.path.join(artist_dir, 'lyrics.txt')
    splits_dir = os.path.join(artist_dir, 'splits')

    try:
        os.makedirs(splits_dir)
    except FileExistsError:
        return

    train_path = os.path.join(splits_dir, 'train.txt')
    val_path = os.path.join(splits_dir, 'val.txt')
    test_path = os.path.join(splits_dir, 'test.txt')

    with open(lyrics_path, 'r') as flyrics,     \
         open(train_path, 'w') as ftrain,       \
         open(val_path, 'w') as fval,           \
         open(test_path, 'w') as ftest:

        reader = song_reader(flyrics)

        try:
            while True:
                for _ in range(train_ct):
                    song = next(reader)
                    ftrain.write(song + '\n')

                for _ in range(val_ct):
                    song = next(reader)
                    fval.write(song + '\n')

                for _ in range(test_ct):
                    song = next(reader)
                    ftest.write(song + '\n')

        except StopIteration:
            return


def get_lyrics_iterators(artist_dir, batch_sizes=(5, 5, 5)):
    split_songs(artist_dir)
    corpus = data.Corpus(artist_dir)

    #print(corpus.dictionary.word2phonemes['raptorspaymybills'])
    #exit()

    splits_dir = os.path.join(artist_dir, 'splits')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create fields
    text_tokenize = lambda x: re.findall(r'\S+|\n',x)
    TEXT = Field(sequential=True, tokenize=text_tokenize, lower=True, include_lengths=True, batch_first=True)

    phoneme_tokenize = lambda x: x.split()
    PHONEME = Field(sequential=True, tokenize=phoneme_tokenize, lower=False, batch_first=True)
    PHONEMES = NestedField(PHONEME, include_lengths=True)


    fields = [('text', TEXT), ('phonemes', PHONEMES)]

    train_ds, val_ds, test_ds = StanzaDataset.splits(
            path=splits_dir, train='train.txt', validation='val.txt', test='test.txt',
            format='tsv', fields=fields, corpus=corpus)

    TEXT.build_vocab(train_ds)
    PHONEMES.build_vocab(train_ds)

    #print(PHONEMES.vocab.stoi)
    #exit()

    # TODO not needed anymore?
    corpus.build_vocab_to_corpus_idx_translate(TEXT.vocab)

    # Create dataloaders
    train_it, val_it, test_it = BucketIterator.splits(
        (train_ds, val_ds, test_ds),
        batch_sizes=batch_sizes,
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        repeat=False,
    )

    return corpus, TEXT.vocab, PHONEMES.vocab, train_it, val_it, test_it


def get_dataloader(artist_dir, batch_sizes=(5, 5, 5)):
    corpus, txt_vocab, phoneme_vocab, train_it, val_it, test_it = get_lyrics_iterators(artist_dir, batch_sizes)

    dataloader = {'train': train_it, 'val': val_it, 'test': test_it}
    return dataloader, txt_vocab, phoneme_vocab



if __name__ == '__main__':
    artist_dir = '../data/lyrics/KendrickLamar'

    split_songs(artist_dir)

    c, txt_vocab, p_vocab, train_it, val_it, test_it = get_lyrics_iterators(artist_dir)
    d = c.dictionary

    #vidx = vocab.stoi['the']
    #print(c.vocab_word_to_phoneme_idxs(vidx))

    for i, batch in enumerate(val_it):
        text, text_lengths = batch.text
        phonemes, phonemes_chunk_lengths, phoneme_lengths = batch.phonemes

        stanza = text[-1]

        phoneme_stanza = phonemes[-1]
        phoneme_stanza_lengths = phoneme_lengths[-1]

        #words = ' '.join([txt_vocab.itos[idx] for i, idx in enumerate(stanza) if i < text_lengths[-1]])
        words = ' '.join([txt_vocab.itos[idx] for i, idx in enumerate(stanza)])
        print(words)
        print()

        print(text.size())
        print(phonemes.size())
        print(phoneme_lengths.size())

        exit()

        #print()
        print(text_lengths)
        print(phonemes_chunk_lengths)

        print([' '.join([p_vocab.itos[idx] for i, idx in enumerate(pword) if i < pword_len])
            for pword, pword_len
            in zip(phoneme_stanza, phoneme_stanza_lengths)])

        #phonemes = [c.vocab_word_to_phoneme_idxs(idx.item()) for idx in stanza]
        #print(phonemes)

        #print()
        exit()
