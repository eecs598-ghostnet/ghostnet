import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from tqdm import tqdm

from dataloader import *

def train(artist_dir, num_epochs=100):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO add multiple artists
    corpus, vocab, train_it = get_lyrics_iterator(artist_dir)


    # TODO actually add a model lol. Probably need to pass corpus into the
    # model so it can retrieve phonemes for a word with
    # `corpus.vocab_word_to_phoneme_idxs`

    # model = ???
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs}')

        running_loss = 0.0

        for i, batch in tqdm(enumerate(train_it)):
            stanzas = batch.text.transpose(0,1) # transpose to batch first



if __name__=='__main__':
    artist_dir = '../data/lyrics/KendrickLamar'
    train(artist_dir, num_epochs=3)
