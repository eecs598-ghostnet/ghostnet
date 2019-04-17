import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from text_generation_model import *
from dataloader import get_dataloader
from gen_text import load_model
import utils
import config


def train_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
                # text is size (batch_size, max_stanza_length)
                # text_lengths is size (batch_size,); contains lengths of each stanza in batch
                text, text_lengths = batch.text
                text = text.to(device)
                text_lengths = text_lengths.to(device)

                # phonemes is size (batch_size, max_stanza_length, max_word_length)
                # 2nd arg should be same as text_lengths
                # phoneme_lengths is (batch_size, max_stanza_length); x[i][j] contains word length of word j in the ith stanza
                phonemes, _, phoneme_lengths = batch.phonemes
                phonemes = phonemes.to(device)
                phoneme_lengths = phoneme_lengths.to(device)

                # TODO probably need to trim text and phonemes into `input` and `labels`
                # Should already be on device but worth checking
                text_inputs = text[:, :-1] # Offset stanzas by one word.
                labels = text[:, 1:]
                text_lengths = text_lengths - 1
                text_inputs = text_inputs.to(device)
                labels = labels.to(device)
#                labels = labels.to(float)

                phoneme_inputs = phonemes[:, :-1, :] # Get rid of last timestep to match text inputs
                phoneme_lengths = phoneme_lengths[:, :-1]
                phoneme_inputs = phoneme_inputs.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #outputs = model(text_inputs, phoneme_inputs, text_lengths, phoneme_lengths)
                    #outputs = outputs.reshape(labels.shape[0] * labels.shape[1], -1)
                    outputs, loss = model(text_inputs, phoneme_inputs,
                                          text_lengths, phoneme_lengths,
                                          labels)

                    preds = outputs.argmax(dim=-1)


                # backward + optimize only if in training phase
                if phase == 'train':
                    #loss = criterion(outputs, labels)

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
                running_loss += loss.item() * text_inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))

    utils.show_plot(range(len(loss_history)), loss_history)
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
    model_params = config.get_model_params(txt_vocab, phoneme_vocab)

    ############### Load model from scratch
    #model = TextGenerationModel(**model_params)

    ## Load pretrained embeddings for text vocab
    #print('Loading vectors...')
    #txt_vocab.load_vectors('glove.6B.100d')
    #model.embedding.weight.data.copy_(txt_vocab.vectors)
    #model = model.to(device)
    ##############


    ############## Load saved model
    model = load_model('../model/adaptive_softmax_10.pt', device, **model_params)
    ##############


    ############## Load AE pretraining
    #AE_path = '../model/autoencoder/phoneme_auto_test.pt'
    #ae_params = config.get_ae_params(phoneme_vocab)
    #ae = PhonemeAutoencoder(**ae_params)

    #ae.load_state_dict(torch.load(AE_path, map_location=device))

    ## Load into model
    #model.phoneme_embedding.load_state_dict(ae.phoneme_embedding.state_dict())
    #model.phoneme_encoder.load_state_dict(ae.encoder.state_dict())

    # freeze weights as a test TODO
    #for param in model.phoneme_encoder.parameters():
    #    param.requires_grad = False
    ##############


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    model = train_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, exp_lr_scheduler, num_epochs=30)
    torch.save(model.state_dict(), '../model/adaptive_softmax_30.pt')


if __name__ == '__main__':
    main()
