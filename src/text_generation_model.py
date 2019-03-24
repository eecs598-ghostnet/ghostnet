import torch
import torch.nn as nn
import torch.optim as optim


class TextGenerationModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, phoneme_vocab_size, phoneme_embed_size, phoneme_hidden_size):
        super(TextGenerationModel, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.phoneme_vocab_size = phoneme_vocab_size
        self.phoneme_embed_size = phoneme_embed_size
        self.phoneme_hidden_size = phoneme_hidden_size

        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, phoneme_embed_size)
        self.phoneme_lstm = nn.LSTM(self.phoneme_embed_size, self.phoneme_hidden_size, batch_first=True)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def _initialize_hidden(self, batch_size):
        phoneme_hidden = Variable(torch.zeros(batch_size, self.phoneme_hidden_size), requires_grad=False).double()
        phoneme_cell = Variable(torch.zeros(batch_size, self.phoneeme_hidden_size), requires_grad=False).double()
        hidden = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False).double()
        cell = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False).double()
        return (phoneme_hidden, phoneme_cell), (hidden, cell)

    def forward(self, x, x_phonemes, lengths, phoneme_lengths):
        """ 
        Forward function for the text generation model.

        Inputs:
        - x, Pytorch variable with inputs of size (batch_size, T) where each value is the index of a
          word.
        - x_phonemes, Pytorch variable with input phonemes of size (batch_size, T, T_phoneme).
        - length, Pytorch tensor of sample lengths of shape (batch_size,).

        Returns:
        - outputs, Pytorch variabl of size (batch_size, T, vocab_size) containing distribution over
          possible next words for each time step in each sequence.
        """
        B, T = x.shape
        _, _, T_phoneme = x_phonemes.shape
        embeddings = self.embedding(x)
        phoneme_hiddens, hiddens = self._initialize_hidden(B)
        #TODO : Need to use pack padded here.
        outputs = []
        for t in range(T):
            outputs_t, hiddens = self.lstm(x, hiddens)
            outputs.append(outputs_t)
        outputs = torch.concatenate(outputs, dim=1)
        outputs = self.fc(outputs.view(B, -1))
        outputs = self.softmax(outputs)
        outputs = outputs.view(B, T, self.vocab_size)
        return outputs

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding_layer = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, inputs, hidden):
        embeddings = self.embedding_layer(inputs)
        output, hidden = self.lstm(embeddings)

        return output

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden):
        embeddings = self.embedding(inputs)
        outputs, hiddens = self.lstm(embeddings)
        outputs = self.out(outputs)
        outputs = self.softmax(outputs)
        return outputs


def main():
    vocab_size = 1000
    embed_size = 50
    hidden_size = 100
    phoneme_vocab_size = 60
    phoneme_embed_size = 30
    phoneme_hidden_size = 40
    
    model = TextGenerationModel(vocab_size, embed_size, hidden_size, phoneme_vocab_size, phoneme_embed_size, phoneme_hidden_size)

if __name__ == '__main__':
    main()
