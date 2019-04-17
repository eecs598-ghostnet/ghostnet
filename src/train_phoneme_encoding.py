import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from text_generation_model import PhonemeAutoencoder
from dataloader import get_dataloader
from gen_text import load_model
import utils
import config


def train_model(device, dataloaders, dataset_sizes, model, criterion,
                optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())

    loss_history = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and a validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch in dataloaders[phase]:
                # phonemes is size (batch_size, max_stanza_length, max_word_length)
                # 2nd arg should be same as text_lengths
                # phoneme_lengths is (batch_size, max_stanza_length); x[i][j] contains word length of word j in the ith stanza
                phonemes, _, phoneme_lengths = batch.phonemes
                phonemes = phonemes.to(device)
                phoneme_lengths = phoneme_lengths.to(device)

                B, T, T_phoneme = phonemes.size()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(phonemes, phoneme_lengths)

                labels = phonemes.view(B*T*T_phoneme)
                outputs = outputs.view(B*T*T_phoneme, -1).to(device)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss = criterion(outputs, labels)

                    try:
                        # Free cached gpu mem
                        del outputs
                        torch.cuda.empty_cache()
                    except RuntimeError:
                        pass

                    loss.backward()
                    optimizer.step()
                    loss_history.append(loss.item())

                print('loss: ', loss.item())

                # statistics
                running_loss += loss.item() * B

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = 0.0
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val':
                best_model_wts = copy.deepcopy(model.state_dict())

    #time_elapsed = time.time() - since
    #print('Training complete in {:.0f}m {:.0f}s'.format(
    #    time_elapsed // 60, time_elapsed % 60))
    #print('Best val Acc: {:.4f}'.format(best_acc))

    #utils.show_plot(range(len(loss_history)), loss_history)
    # load best model weights
    if num_epochs > 0:
        model.load_state_dict(best_model_wts)
    return model


def main():
#    vocab_size = 1000
#    embed_size = 50
#    hidden_size = 100
#    phoneme_vocab_size = 60
#    phoneme_embed_size = 30
#    phoneme_hidden_size = 40


    artist_dir = '../data/lyrics/combined_trunc'

    # Get dataloaders
    dataloaders, txt_vocab, phoneme_vocab, _ = get_dataloader(
        artist_dir, batch_sizes=(12, 5, 5), min_vocab_freq=3
    )
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Model
    ae_params = config.get_ae_params(phoneme_vocab)

    # TODO
    model = PhonemeAutoencoder(**ae_params)

    ## Load pretrained embeddings for text vocab
    #txt_vocab.load_vectors('glove.6B.100d')
    #model.embedding.weight.data.copy_(txt_vocab.vectors)

    model = model.to(device)
    ###

    # TODO load previous model state
    #model = load_model('../model/stacked_lstm_25.pt', device, **model_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, exp_lr_scheduler, num_epochs=15)
    torch.save(model.state_dict(), '../model/autoencoder/phoneme_auto_test.pt')


if __name__ == '__main__':
    main()
