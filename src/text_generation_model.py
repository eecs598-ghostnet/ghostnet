import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
        self.lstm = nn.LSTM(embed_size + phoneme_hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def _initialize_hidden(self, batch_size):
        phoneme_hidden = Variable(torch.zeros(batch_size, self.phoneme_hidden_size), requires_grad=False).double()
        phoneme_cell = Variable(torch.zeros(batch_size, self.phoneme_hidden_size), requires_grad=False).double()
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
        - lengths, Pytorch tensor of sample lengths of shape (batch_size,).
        - phoneme_lengths, Pytorch tensor of sample word phoneme lengths of shape (batch_size, T).

        Returns:
        - outputs, Pytorch variable of size (batch_size, T, vocab_size) containing distribution over
          possible next words for each time step in each sequence.
        """
        B, T = x.shape
        _, _, T_phoneme = x_phonemes.shape
        embeddings = self.embedding(x)
        phoneme_embeddings = self.phoneme_embedding(x_phonemes.view(B, -1)).view(B, T, T_phoneme, -1)
        #phoneme_hiddens, hiddens = self._initialize_hidden(B)
        #TODO : Need to use pack padded here.
        #outputs = []
        #TODO: Use pack padded for phoneme sequence so we don't include hiddens after padding.
        phoneme_lengths = phoneme_lengths.contiguous().view(-1)
        phoneme_embeddings = phoneme_embeddings.view(B * T, T_phoneme, -1)
        sorted_phoneme_lengths, argsort_phoneme_lengths = phoneme_lengths.sort(descending=True)
        sorted_phoneme_embeddings = phoneme_embeddings[argsort_phoneme_lengths]
        sorted_phoneme_embeddings = sorted_phoneme_embeddings
        if min(sorted_phoneme_lengths).item() == 0:
            #phoneme_pad_start = torch.argmin(sorted_phoneme_lengths)
            phoneme_pad_start = np.argmin(sorted_phoneme_lengths.cpu().numpy())
        else:
            phoneme_pad_start = len(sorted_phoneme_lengths)
        sorted_phoneme_embeddings_data = sorted_phoneme_embeddings[:phoneme_pad_start,:,:]
        sorted_phoneme_embeddings_pad = sorted_phoneme_embeddings[phoneme_pad_start:,:,:]
        try:
            packed_phoneme_embeddings = pack_padded_sequence(sorted_phoneme_embeddings_data, sorted_phoneme_lengths[:phoneme_pad_start], batch_first=True)
        except:
            import pdb; pdb.set_trace()
        packed_phoneme_outputs, phoneme_hiddens = self.phoneme_lstm(packed_phoneme_embeddings)
        sorted_phoneme_pad_outputs, _ = self.phoneme_lstm(sorted_phoneme_embeddings_pad)
        sorted_phoneme_pad_outputs = sorted_phoneme_pad_outputs[:, :phoneme_lengths.max(), :]

        sorted_phoneme_outputs, _ = pad_packed_sequence(packed_phoneme_outputs, batch_first=True)
        if sorted_phoneme_outputs.shape[1] != sorted_phoneme_pad_outputs.shape[1]:
            import pdb; pdb.set_trace()
        try:
            sorted_phoneme_outputs = torch.cat([sorted_phoneme_outputs, sorted_phoneme_pad_outputs], dim=0)
        except:
            import pdb; pdb.set_trace()
        _, unargsort_phoneme_lengths = argsort_phoneme_lengths.sort()
        phoneme_outputs = sorted_phoneme_outputs[unargsort_phoneme_lengths]
#        idx = (torch.LongTensor(phoneme_lengths) - 1).clamp(min=0).view(-1, 1).expand(len(phoneme_lengths), phoneme_outputs.size(2))
        idx = (phoneme_lengths.long() - 1).clamp(min=0).view(-1, 1).expand(len(phoneme_lengths), phoneme_outputs.size(2))
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)
        last_phoneme_outputs = phoneme_outputs.gather(time_dimension, idx).squeeze(time_dimension)
        last_phoneme_outputs = last_phoneme_outputs.view(B, T, -1)


        ############
        #_, phoneme_hiddens = self.phoneme_lstm(packed_phoneme_embeddings.view(B * T, T_phoneme, -1))
        #phoneme_hiddens = phoneme_hiddens[0].view(B, T, -1)
        # We now have embeddings (B, T, self.embed_size) and phoneme_hiddens (B, T, self.phoneme_hidden_size)
        # which we  will concatenate into the text generation LSTM for input.
        inputs = torch.cat([embeddings, last_phoneme_outputs], dim=2)
        #TODO : Need to use pack padded here.
        outputs, hiddens = self.lstm(inputs)

#        for t in range(T):
#            embeddings_t = embeddings[:, t]
#            phoneme_embeddings_t = phoneme_embeddings[:, t, :, :]
#            #TODO: Use pack padded for phoneme sequence.
#            phoneme_outputs_t, _ = self.phoneme_lstm(phoneme_embeddings_t)
#            inputs_t  = torch.concat([embeddings_t, phoneme_outputs_t], dim=1)
#            outputs_t, hiddens = self.lstm(x_t, hiddens)
#            outputs.append(outputs_t)
#        outputs = torch.concatenate(outputs, dim=1)
        outputs = self.fc(outputs.contiguous().view(B * T, -1))
#        outputs = self.softmax(outputs)
        outputs = outputs.view(B, T, self.vocab_size)
        return outputs

    def gen_word(self, txt, phonemes, hidden=None):
        """
        Predict the next word. Similar to forward but runs on single inputs
        rather than batches and has word sequence length 1.

        Inputs:
        - txt, Pytorch Tensor with size (1)
        - phonemes, Pytorch Tensor with size (1, 1, T_phoneme)
        """
        T_phoneme = phonemes.size(2)

        txt_embedding = self.embedding(txt)
        phoneme_embeddings = self.phoneme_embedding(phonemes.view(1,-1)).view(1, T_phoneme, -1)

        # Phoneme summarization
        phoneme_summ, _ = self.phoneme_lstm(phoneme_embeddings)
        phoneme_summ = phoneme_summ.view(1, 1, T_phoneme, -1)[:,:,-1,:]   # unnecessary but consistent

        txt_gen_inputs = torch.cat((txt_embedding, phoneme_summ), dim=2)

        if hidden is None:
            outputs, hidden = self.lstm(txt_gen_inputs)
        else:
            outputs, hidden = self.lstm(txt_gen_inputs, hidden)

        outputs = self.fc(outputs.contiguous().view(1, -1))
        outputs = outputs.view(self.vocab_size)     # only predicting one word
        return outputs, hidden


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

    batch_size = 2
    T = 10
    T_phoneme = 3
    dummy_x = np.random.choice(vocab_size, (batch_size, T))
    dummy_x_phonemes = np.random.choice(phoneme_vocab_size, (batch_size, T, T_phoneme))
    dummy_lengths = np.random.choice(T, (batch_size,))
    dummy_phoneme_lengths = np.random.choice(np.array(range(1,T_phoneme+1)), (batch_size, T))

    # Convert to pytorch variables
    dummy_x = torch.from_numpy(dummy_x)
    dummy_x_phonemes = torch.from_numpy(dummy_x_phonemes)
    dummy_lengths = torch.from_numpy(dummy_lengths)
    dummy_phoneme_lengths = torch.from_numpy(dummy_phoneme_lengths)
    dummy_outputs = model(dummy_x, dummy_x_phonemes, dummy_lengths, dummy_phoneme_lengths)

    for b in range(batch_size):
        sample_length = dummy_lengths[b]
        dummy_x[b, sample_length:] = 0.0
        for t in range(T):
            word_phoneme_length = dummy_phoneme_lengths[b, t]
            dummy_x_phonemes[b, t, word_phoneme_length:] = 0.0

    dummy_outputs = model(dummy_x, dummy_x_phonemes, dummy_lengths, dummy_phoneme_lengths)
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    main()
