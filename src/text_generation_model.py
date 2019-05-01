import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PhonemeEncoder(nn.Module):
    def __init__(self, phoneme_embed_size, phoneme_hidden_size):
        super().__init__()
        self.phoneme_embed_size = phoneme_embed_size
        self.phoneme_hidden_size = phoneme_hidden_size
        self.phoneme_lstm_encoder = nn.LSTM(phoneme_embed_size, phoneme_hidden_size, batch_first=True)

    def forward(self, phoneme_embeddings, phoneme_lengths):
        """
        Encode phoneme sequences to a fixed length representation.

        Inputs:
        - phonemes_embeddings, size (B, T, T_phoneme, phoneme_embed_size)
        - phoneme_lengths, size (B, T)

        Outputs:
        - phoneme_summaries, size (B, T, phoneme_hidden_size)
        """
        B, T, T_phoneme, _ = phoneme_embeddings.size()

        # Get phoneme embeddings -> (B, T, T_phoneme, phoneme_embed_size)
        #phoneme_embeddings = self.phoneme_embedding(phonemes.view(B, -1)).view(B, T, T_phoneme, -1)

        # Summarize phoneme embeddings
        # Unravel to sequence of pwords=(T_phoneme, phoneme_embed_size)
        phoneme_lengths = phoneme_lengths.contiguous().view(-1)
        phoneme_embeddings = phoneme_embeddings.view(B * T, T_phoneme, -1)

        # Sort phonemes sequence by word-length-in-phonemes
        sorted_phoneme_lengths, argsort_phoneme_lengths = phoneme_lengths.sort(descending=True)
        sorted_phoneme_embeddings = phoneme_embeddings[argsort_phoneme_lengths]
        #print(f'sorted_phoneme_lengths:\n{sorted_phoneme_lengths}')

        # Split phonemes into actual words and pads
        if min(sorted_phoneme_lengths).item() == 0:
            #phoneme_pad_start = torch.argmin(sorted_phoneme_lengths)
            phoneme_pad_start = np.argmin(sorted_phoneme_lengths.cpu().numpy())
        else:
            phoneme_pad_start = len(sorted_phoneme_lengths)

        sorted_phoneme_embeddings_data = sorted_phoneme_embeddings[:phoneme_pad_start,:,:]
        sorted_phoneme_embeddings_pad = sorted_phoneme_embeddings[phoneme_pad_start:,:,:]

        # Summarize words -> (B*T, phoneme_hidden)
        packed_phoneme_embeddings = pack_padded_sequence(
            sorted_phoneme_embeddings_data,
            sorted_phoneme_lengths[:phoneme_pad_start],
            batch_first=True
        )
        packed_phoneme_outputs, _ = self.phoneme_lstm_encoder(packed_phoneme_embeddings)
        sorted_phoneme_outputs, _ = pad_packed_sequence(packed_phoneme_outputs, batch_first=True)
        #print(sorted_phoneme_outputs.size())

        # If pads, run through lstm and cat with data outputs
        if phoneme_pad_start != len(sorted_phoneme_lengths):
            # Does this matter at all? Running LSTM on sequences of length 0
            sorted_phoneme_pad_outputs, _ = self.phoneme_lstm_encoder(sorted_phoneme_embeddings_pad)
            sorted_phoneme_pad_outputs = sorted_phoneme_pad_outputs[:, :phoneme_lengths.max(), :]

            sorted_phoneme_outputs = torch.cat([sorted_phoneme_outputs, sorted_phoneme_pad_outputs], dim=0)


        # Unsort phoneme summaries
        _, unargsort_phoneme_lengths = argsort_phoneme_lengths.sort()
        phoneme_outputs = sorted_phoneme_outputs[unargsort_phoneme_lengths]
        #print(f'phoneme_outputs size: {phoneme_outputs.size()}')

        # Get the last non-pad hidden state of each phoneme sequence
#        idx = (torch.LongTensor(phoneme_lengths) - 1).clamp(min=0).view(-1, 1).expand(len(phoneme_lengths), phoneme_outputs.size(2))
        idx = (phoneme_lengths.long() - 1).clamp(min=0).view(-1, 1).expand(len(phoneme_lengths), phoneme_outputs.size(2))
        time_dimension = 1
        idx = idx.unsqueeze(time_dimension)
        last_phoneme_outputs = phoneme_outputs.gather(time_dimension, idx).squeeze(time_dimension)
        last_phoneme_outputs = last_phoneme_outputs.view(B, T, -1)

        return last_phoneme_outputs


class PhonemeDecoder(nn.Module):
    def __init__(self, phoneme_embed_size, phoneme_hidden_size, phoneme_vocab_size):
        super().__init__()
        self.phoneme_hidden_size = phoneme_hidden_size
        self.phoneme_vocab_size = phoneme_vocab_size

        self.decoder_lstm = nn.LSTM(phoneme_embed_size, phoneme_hidden_size, batch_first=True)
        self.fc = nn.Linear(phoneme_hidden_size, phoneme_vocab_size)

    def forward(self, phoneme_embedding, h_0, c_0=None):
        """
        Runs one timestep of decoder, returns scores, hidden.

        Inputs:
        - phoneme_embeddings, size (B*T, 1, embed_dim)
        - h_0, c_0, size (1, B*T, hidden_dim)
        """
        _, BT, phoneme_hidden_size = h_0.size()

        if c_0 is None:
            c_0 = torch.zeros_like(h_0)

        h, hidden = self.decoder_lstm(phoneme_embedding, (h_0, c_0))

        h = h.view(BT, self.phoneme_hidden_size)    # (BT, 1, hidden) -> ..

        scores = self.fc(h)     # scores over phoneme vocabulary
        return scores, hidden


class PhonemeAutoencoder(nn.Module):
    def __init__(self, phoneme_vocab_size, phoneme_embed_size, phoneme_hidden_size, sos_token):
        super().__init__()
        self.sos_token = sos_token
        self.phoneme_vocab_size = phoneme_vocab_size
        self.phoneme_embed_size = phoneme_embed_size
        self.phoneme_hidden_size = phoneme_hidden_size

        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, phoneme_embed_size)
        self.encoder = PhonemeEncoder(phoneme_embed_size, phoneme_hidden_size)
        self.decoder = PhonemeDecoder(phoneme_embed_size, phoneme_hidden_size, phoneme_vocab_size)

    def forward(self, phonemes, phoneme_lengths):
        """
        Ipnuts:
        - phonemes, size (B, T, T_phoneme)
        - phoneme_lengths, size (B, T)

        Returns scores of shape (B, T, T_phoneme, phoneme_vocab_size)
        """
        B, T, T_phoneme = phonemes.size()
        device = phonemes.device

        # Encode phonemes
        phoneme_embeddings = self.phoneme_embedding(phonemes)   # (B,T,T_p, embed_dim)
        phoneme_encodings = self.encoder(phoneme_embeddings, phoneme_lengths)

        # Decode step-by-step into phonemes (scores to be softmaxed) again
        h, c = phoneme_encodings.view(1, B*T, self.phoneme_hidden_size), None
        phoneme_scores = torch.zeros((B*T, T_phoneme, self.phoneme_vocab_size))
        token = torch.ones((B*T, 1)).to(device).long() * self.sos_token   # init with <sos>

        for t in range(T_phoneme):
            gen_phoneme_embedding = self.phoneme_embedding(token).view(B*T, 1, -1)

            # scores_t is shape (B*T, phoneme_vocab_size)
            scores_t, (h,c) = self.decoder(gen_phoneme_embedding, h, c)
            phoneme_scores[:,t,:] = scores_t

            # Feed max phoneme back into decoder
            token = torch.argmax(scores_t, -1, keepdim=True)

        phoneme_scores.view(B, T, T_phoneme, self.phoneme_vocab_size)

        return phoneme_scores


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)

    def forward(self, inputs, lengths):
        """
        Return the hidden states for each timestep of inputs.
        Inputs:
        - inputs: size (B, T, input_size)

        Outputs:
        - outputs: size (B, T, hidden_size)
        """
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True)

        packed_outputs, hiddens = self.lstm(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs

class Decoder(nn.Module):
    """
    Decoder in this case is just doing one word prediction from a sequence of
    encoder weights. Prediction is done in next layer, this outputs same
    hidden_size which will then be transformed at final prediction layer.
    """
    def __init__(self, input_size, hidden_size, max_length):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length

        self.attn = nn.Linear(hidden_size+input_size, max_length)
        self.attn_combine = nn.Linear(hidden_size+input_size, hidden_size)

        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden, encodings):
        prev_h = hidden[0].squeeze(0)

        # Get attn weights from input (embedding+phoneme_embedding of last word), hidden
        attn_weights = self.attn(torch.cat((input, prev_h), dim=1))
        attn_weights = F.softmax(attn_weights, dim=1)

        # Get weighted sum of encoder outputs of size (B, 1, hidden_size)
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(1),
            encodings
        )

        # Combine applied attention with embeddings (ctx + phnm) of prev word
        combined = torch.cat((attn_applied.squeeze(1), input), dim=1)
        attn_combined = F.relu(self.attn_combine(combined)) # (B, hidden_size)

        output, hidden = self.lstm(attn_combined.unsqueeze(1), hidden)

        return output, hidden



class AttentionEncoderDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, phoneme_vocab_size,
                 phoneme_embed_size, phoneme_hidden_size, max_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.phoneme_vocab_size = phoneme_vocab_size
        self.phoneme_embed_size = phoneme_embed_size
        self.phoneme_hidden_size = phoneme_hidden_size
        self.max_length = max_length

        self.phoneme_embedding = nn.Embedding(phoneme_vocab_size, phoneme_embed_size)
        self.phoneme_dropout = nn.Dropout()
        self.phoneme_encoder = PhonemeEncoder(phoneme_embed_size, phoneme_hidden_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_dropout = nn.Dropout()

        self.encoder = EncoderRNN(embed_size+phoneme_hidden_size, hidden_size)
        self.decoder = Decoder(embed_size+phoneme_hidden_size, hidden_size, max_length)

        # Adaptive Softmax includes fc layers with size dependent on bucket
        self.adaptivesoftmax = nn.AdaptiveLogSoftmaxWithLoss(
            hidden_size, vocab_size, cutoffs=[
                # First bucket is special tokens (unk, pad, sos, eos, \n)
                5, 50, 500, 5000,
            ], div_value=2.0,
        )

    def forward(self, x, x_phonemes, lengths, phoneme_lengths, targets):
        device = x.device
        B, T = x.shape
        _, _, T_phoneme = x_phonemes.shape

        # Get word embeddings -> (B, T, word_embed_size)
        embeddings = self.embedding(x)
        embeddings = self.embed_dropout(embeddings)

        # Get phoneme embeddings -> (B, T, T_phoneme, phoneme_embed_size)
        phoneme_embeddings = self.phoneme_embedding(x_phonemes.view(B, -1)).view(B, T, T_phoneme, -1)
        phoneme_embeddings = self.phoneme_dropout(phoneme_embeddings)
        last_phoneme_outputs = self.phoneme_encoder(phoneme_embeddings, phoneme_lengths)

        inputs = torch.cat([embeddings, last_phoneme_outputs], dim=2)

        # Get encodings for each timestep
        encodings = self.encoder(inputs, lengths)

        # Init (h, c), encodings for applying attention
        decoder_hidden = (torch.zeros((1 ,B, self.hidden_size), device=device),
                          torch.zeros((1 ,B, self.hidden_size), device=device))
        attn_encodings = torch.zeros((B, self.max_length, self.hidden_size), device=device)
        attn_encodings[:, :T, :] = encodings # TODO XXX

        # Prediction at each timestep
        outputs = []
        for t in range(T):
            decoder_input = inputs[:, t, :].view(B, -1)

            # Get output of size (B, [1], hidden_size)
            # XXX
            attn_mask = torch.zeros_like(attn_encodings)
            attn_mask[:, :t, :] = 1.0
            partial_encodings = attn_encodings * attn_mask

            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, partial_encodings)
            outputs.append(output.squeeze(1))


        outputs = torch.stack(outputs, dim=1)


        # Mask outputs where padded because we are doing loss here
        mask = torch.zeros_like(targets, device=device)
        for n in range(mask.shape[0]):
            length = lengths[n]
            mask[n, :length] = 1.0    # offset for sos and eos



        outputs = outputs.contiguous().view(B*T, -1)
        targets = targets.contiguous().view(B*T).squeeze()

        mask = mask.contiguous().view(B*T).squeeze()

        targets = targets[mask.byte()].clone()
        outputs = outputs[mask.byte()].clone()


        # Get loss and scores from adaptive softmax
        import pdb; pdb.set_trace()
        outputs, loss = self.adaptivesoftmax(outputs, targets)

        # TODO need to 'repad' output for correct seqs? Right now unfolded to
        # one long sequence and padding removed
        #outputs = outputs.view(B, T)

        return outputs, loss


    def gen_word(self, txt, phonemes, encodings, encoder_hidden=None, decoder_hidden=None):
        """
        Predict the next word. Similar to forward but runs on single inputs
        rather than batches and has word sequence length 1.

        Inputs:
        - txt, Last word generated so far with size (1)
        - phonemes, Pytorch Tensor with size (1, 1, T_phoneme)
        - encodings, list of h0 outputs from encoder on sentence generated so far
        - encoder_hidden, hidden state (h, c) of encoder
        - decoder_hidden, hidden state (h, c) of decoder
        """
        device = txt.device
        T_phoneme = phonemes.size(2)

        txt_embedding = self.embedding(txt)
        phoneme_embeddings = self.phoneme_embedding(phonemes.view(1,-1)).view(1, T_phoneme, -1)

        # Phoneme summarization
        phoneme_summ, _ = self.phoneme_encoder.phoneme_lstm_encoder(phoneme_embeddings)
        phoneme_summ = phoneme_summ.view(1, 1, T_phoneme, -1)[:,:,-1,:]   # unnecessary but consistent

        inputs = torch.cat((txt_embedding, phoneme_summ), dim=2)

        # Take one step of encoderRNN -> (1, 1, hidden_size)
        encoding, encoder_hidden = self.encoder.lstm(inputs, encoder_hidden)
        encodings.append(encoding)


        # Construct decoder inputs
        if decoder_hidden is None:
            decoder_hidden = (torch.zeros((1 ,1, self.hidden_size), device=device),
                              torch.zeros((1 ,1, self.hidden_size), device=device))
        decoder_input = inputs[:, -1, :].view(1, -1)
        attn_encodings = torch.zeros((1, self.max_length, self.hidden_size), device=device)
        for t, encoding in enumerate(encodings):
            attn_encodings[:, t, :] = encodings[t]

        outputs, decoder_hidden = self.decoder(decoder_input, decoder_hidden, attn_encodings)

        #####
        outputs = outputs.contiguous().view(1, -1)
        outputs = self.adaptivesoftmax.log_prob(outputs)

        outputs = outputs.view(self.vocab_size)
        return outputs, encodings, encoder_hidden, decoder_hidden


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
        #self.phoneme_lstm = nn.LSTM(self.phoneme_embed_size, self.phoneme_hidden_size, batch_first=True)
        self.phoneme_encoder = PhonemeEncoder(phoneme_embed_size, phoneme_hidden_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + phoneme_hidden_size, hidden_size, num_layers=2, batch_first=True)

        #self.fc = nn.Linear(hidden_size, vocab_size)

        # Adaptive Softmax includes fc layers with size dependent on bucket
        self.adaptivesoftmax = nn.AdaptiveLogSoftmaxWithLoss(
            hidden_size, vocab_size, cutoffs=[
                # First bucket is special tokens (unk, pad, sos, eos, \n)
                # TODO decide cutoffs
                5, 50, 500, 5000,
            ], div_value=4.0,
        )

    def _initialize_hidden(self, batch_size):
        phoneme_hidden = Variable(torch.zeros(batch_size, self.phoneme_hidden_size), requires_grad=False).double()
        phoneme_cell = Variable(torch.zeros(batch_size, self.phoneme_hidden_size), requires_grad=False).double()
        hidden = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False).double()
        cell = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=False).double()
        return (phoneme_hidden, phoneme_cell), (hidden, cell)

    def forward(self, x, x_phonemes, lengths, phoneme_lengths, targets):
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
        device = x.device
        B, T = x.shape
        _, _, T_phoneme = x_phonemes.shape
        #print(f'B: {B}, T: {T}, T_phoneme: {T_phoneme}')

        # Get word embeddings -> (B, T, word_embed_size)
        embeddings = self.embedding(x)

        # Get phoneme embeddings -> (B, T, T_phoneme, phoneme_embed_size)
        phoneme_embeddings = self.phoneme_embedding(x_phonemes.view(B, -1)).view(B, T, T_phoneme, -1)
        last_phoneme_outputs = self.phoneme_encoder(phoneme_embeddings, phoneme_lengths)

        #print(f'last_phoneme_outputs size: {last_phoneme_outputs.size()}')


        # We now have embeddings (B, T, self.embed_size) and phoneme_hiddens (B, T, self.phoneme_hidden_size)
        # which we  will concatenate into the text generation LSTM for input.
        inputs = torch.cat([embeddings, last_phoneme_outputs], dim=2)

        # Pack combined inputs. Lengths are already sorted here
        packed_inputs = pack_padded_sequence(inputs, lengths, batch_first=True)

        packed_outputs, hiddens = self.lstm(packed_inputs)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)


        # Mask outputs because we are doing loss here
        mask = torch.zeros_like(targets)
        for n in range(mask.shape[0]):
            length = lengths[n]
            mask[n, :length] = 1.0

        outputs = outputs.contiguous().view(B*T, -1)
        targets = targets.contiguous().view(B*T).squeeze()
        mask = mask.contiguous().view(B*T).squeeze()

        targets = targets[mask.byte()]
        outputs = outputs[mask.byte()]

        # Get loss and scores from adaptive softmax
        outputs, loss = self.adaptivesoftmax(outputs, targets)

        #outputs = outputs.view(B, T, -1)


        return outputs, loss

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
        phoneme_summ, _ = self.phoneme_encoder.phoneme_lstm_encoder(phoneme_embeddings)
        phoneme_summ = phoneme_summ.view(1, 1, T_phoneme, -1)[:,:,-1,:]   # unnecessary but consistent

        txt_gen_inputs = torch.cat((txt_embedding, phoneme_summ), dim=2)

        if hidden is None:
            outputs, hidden = self.lstm(txt_gen_inputs)
        else:
            outputs, hidden = self.lstm(txt_gen_inputs, hidden)

        #outputs = self.fc(outputs.contiguous().view(1, -1))
        #outputs = outputs.view(self.vocab_size)     # only predicting one word

        outputs = outputs.contiguous().view(1, -1)
        outputs = self.adaptivesoftmax.log_prob(outputs)

        outputs = outputs.view(self.vocab_size)
        return outputs, hidden


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
