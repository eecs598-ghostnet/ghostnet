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

    def __init__(self, path, format, fields, skip_header=False,
                 csv_reader_params={}, **kwargs):
        format = format.lower()
        make_example = {
            'json': Example.fromJSON, 'dict': Example.fromdict,
            'tsv': Example.fromCSV, 'csv': Example.fromCSV}[format]

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
            reader = stanza_reader(f)

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


def get_lyrics_iterator(artist_dir, batch_sizes=(5, 5, 5)):
    splits_dir = os.path.join(artist_dir, 'splits')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenize = lambda x: re.findall(r'\S+|\n',x)

    TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
    fields = [('text', TEXT)]

    train_ds, val_ds, test_ds = StanzaDataset.splits(
            path=splits_dir, train='train.txt', validation='val.txt', test='test.txt',
            format='tsv', fields=fields)

    TEXT.build_vocab(train_ds)

    train_it, val_it, test_it = BucketIterator.splits(
        (train_ds, val_ds, test_ds),
        batch_sizes=batch_sizes,
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        repeat=False,
    )

    corpus = data.Corpus(artist_dir)
    corpus.build_vocab_to_corpus_idx_translate(TEXT.vocab)

    # TODO train/val/test splits
    return corpus, TEXT.vocab, train_it, val_it, test_it


if __name__ == '__main__':
    artist_dir = '../data/lyrics/combined'

    split_songs(artist_dir)

    c, vocab, train_it, val_it, test_it = get_lyrics_iterator(artist_dir)
    d = c.dictionary

    #vidx = vocab.stoi['the']
    #print(c.vocab_word_to_phoneme_idxs(vidx))

    for i, batch in enumerate(train_it):
        stanzas = batch.text.transpose(0,1)
        stanza = stanzas[0]
        print(stanza.size())

        words = ' '.join([vocab.itos[idx] for idx in stanza])
        print(words)

        phonemes = [c.vocab_word_to_phoneme_idxs(idx.item()) for idx in stanza]
        #print(phonemes)

        #print()
        exit()
