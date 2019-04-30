import argparse
import copy
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from text_generation_model import *
from dataloader import get_dataloader
import utils
import config
from transformer_model import create_context_mask, PhonemeTransformer, create_masks

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def generative_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=25, weights_dir='../model', special_tokens=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_val_loss = 10000.0

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
                #text_transformer_mask = create_context_mask(text_inputs, special_tokens)
                opt = Namespace(src_pad = 1, device=device)
                text_transformer_mask = create_context_mask(text_inputs, opt)
                labels = labels.to(device)
#                labels = labels.to(float)
                text_pad_mask = (text_inputs != opt.src_pad).to(device)

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
                    #try:
                    outputs, loss = model(text_inputs, phoneme_inputs,
                                          text_transformer_mask, phoneme_lengths,
                                          labels, text_pad_mask)
                    #except RuntimeError as e:
                    #    print('RUNTIME ERROR:')
                    #    print(e)
                    #    continue

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
                # TODO: Fix accuracy calculation, right now preds and labels are actually different sizes I think.
                running_loss += loss.item() * text_inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))


            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                best_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                if epoch % 5 == 0:
                    torch.save(model.state_dict(), os.path.join(weights_dir, '{}.pt'.format(epoch)))

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
    parser = argparse.ArgumentParser("Argument parser for training GhostNet-Transformer")
    parser.add_argument("-model_weights_path", type=str, help="Path to model weights for loading checkpoint.")
    parser.add_argument("-artist_dir", type=str, help="Path to artist data.",
        default="../data/lyrics/combined_trunc")
    parser.add_argument("-weights_dir", type=str,
        help="Path to save model weights.", default="../model/transformer")
    args = parser.parse_args()

    torch.manual_seed(12345)

    model_weights_path = args.model_weights_path if args.model_weights_path else None
    artist_dir = args.artist_dir
    weights_dir = args.weights_dir

    ModelType = PhonemeTransformer

    # Get dataloaders
    dataloaders, txt_vocab, phoneme_vocab, _ = get_dataloader(
        artist_dir, batch_sizes=(12, 12, 12), min_vocab_freq=1, max_len=256
    )
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
    print(f'train_dataset size: {dataset_sizes["train"]}')

    # Max length needed for non-positional attention
    max_length = max([len(example.text) for x in ['train', 'val', 'test']
                      for example in dataloaders[x].dataset.examples])
    max_length += 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model
    model_params = config.get_transformer_model_params(txt_vocab, phoneme_vocab)
    model_params['max_length'] = max_length


    ############### Load model from scratch
    if model_weights_path is None:
        model = ModelType(**model_params)

        # Load pretrained embeddings for text vocab
        print('Loading vectors...')
        #txt_vocab.load_vectors('glove.6B.100d')
        #model.context_encoder.embed.weight.data.copy_(txt_vocab.vectors)
        model = model.to(device)
    ##############


    ############## Load saved model
    else:
        model = config.load_model(ModelType, model_weights_path, device, **model_params)
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
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)


    special_tokens = {'src_pad': txt_vocab.stoi['<pad>']}

    model = generative_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, exp_lr_scheduler, num_epochs=50, weights_dir=weights_dir, special_tokens=special_tokens)


if __name__ == '__main__':
    main()
