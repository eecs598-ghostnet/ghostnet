import csv
import os
import re
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


def get_lyrics_iterator(path):
    lyrics_path = os.path.join(path, 'lyrics.txt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenize = lambda x: re.findall(r'\S+|\n',x)

    TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
    fields = [('text', TEXT)]

    train_ds = StanzaDataset(path=lyrics_path, format='tsv', fields=fields)

    TEXT.build_vocab(train_ds)

    train_it = BucketIterator(
        train_ds,
        batch_size=5,
        device=device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        repeat=False,
    )


    # TODO train/val/test splits
    # TODO build corpus with phonemes from TEXT.vocab
    return TEXT.vocab, train_it


if __name__ == '__main__':
    lyrics_dir = '../data/lyrics_headers/KendrickLamar'

    vocab, train_it = get_lyrics_iterator(lyrics_dir)
    c = data.Corpus(lyrics_dir)
    d = c.dictionary

    c.build_vocab_to_corpus_idx_translate(vocab)

    #print(len(vocab.itos))      # These are off by 2 bc <pad> and <unk>
    vidx = vocab.stoi['the']
    print(c.vocab_word_to_phoneme_idx(vidx))

    for i, batch in enumerate(train_it):
        stanzas = batch.text.transpose(0,1)
        stanza = stanzas[0]
        print(stanza.size())

        words = ' '.join([vocab.itos[idx] for idx in stanza])
        print(words)

        print()

        exit()
