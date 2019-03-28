import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from text_generation_model import TextGenerationModel
from dataloader import get_dataloader

def train_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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

                # phonemes is size (batch_size, max_stanza_length, max_word_length)
                # 2nd arg should be same as text_lengths
                # phoneme_lengths is (batch_size, max_stanza_length); x[i][j] contains word length of word j in the ith stanza
                phonemes, _, phoneme_lengths = batch.phonemes

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
                    outputs = model(text_inputs, phoneme_inputs, text_lengths, phoneme_lengths)
                #    outputs = outputs.reshape(labels.shape[0] * labels.shape[1], -1)

                _, preds = torch.max(outputs, 2)

                # gather only outputs for non padded sections.
                mask = torch.zeros(labels.shape)
                for n in range(mask.shape[0]):
                    length = text_lengths[n]
                    mask[n, :length] = 1.0

                outputs = outputs[mask.unsqueeze(2).expand_as(outputs).byte()].view(mask.sum().long(), -1)
                labels = labels[mask.byte()]
                preds = preds[mask.byte()]


                # backward + optimize only if in training phase
                if phase == 'train':
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
  
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

    artist_dir = '../data/lyrics/Drake'
    dataloaders, txt_vocab, phoneme_vocab = get_dataloader(artist_dir)
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}
    vocab_size = len(txt_vocab)
    embed_size = 50
    hidden_size = 100
    phoneme_vocab_size = len(phoneme_vocab)
    phoneme_embed_size = 30
    phoneme_hidden_size = 40
    model = TextGenerationModel(vocab_size, embed_size, hidden_size, phoneme_vocab_size, phoneme_embed_size, phoneme_hidden_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    train_model(device, dataloaders, dataset_sizes, model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)


if __name__ == '__main__':
    main()
