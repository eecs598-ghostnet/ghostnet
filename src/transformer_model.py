#!/bin/python3
# This code is an adaptation of the implemmentations of transformers, presented
# in https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

import copy
import math 
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def nopeak_mask(size, opt):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = (torch.from_numpy(np_mask) == 0)
    if opt.device == 0:
        np_mask = np_mask.cuda()
    return np_mask

def create_masks(src, trg, opt):

    src_mask = (src != opt.src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != opt.trg_pad).unsqueeze(-2)
        size = trg.size(1) # get seq_len for matrix
        np_mask = nopeak_mask(size, opt)
        if trg.is_cuda:
            np_mask.cuda()
            trg_mask = trg_mask.cuda()
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask

def create_context_mask(src, opt):

    src_mask = (src != opt.src_pad).unsqueeze(-2)
    size = src.size(1)
    np_mask = nopeak_mask(size, opt)
    if src.is_cuda:
        np_mask = np_mask.cuda()
        src_mask = src_mask.cuda()
    src_mask = src_mask & np_mask

    return src_mask

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                        math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.sin(pos / (10000 ** (2 * (i + 1) / d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        # add constant to embedding
        seq_len = x.size(1)
        if torch.cuda.is_available():
            x = x + (self.pe[:,:seq_len]).cuda()
        else:
            x = x + (self.pe[:,:seq_len])
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        # We set d_ff as default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()

        self.d_model = d_model
        assert d_model % heads == 0, "d_model must be divisible by heads."
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        bs = q.size(0)

        # perform linear operation and split into heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)


    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class PhonemeEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)

    def forward(self, x):
        """ This class doesn't need mask because we only care about looking
        at all the phonemes for a word.
        """
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, x2, x2))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x_2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, max_length):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_length)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class PhonemeEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)

class TransformerDecoder(nn.Module):
    """ This is a modification of the transformer architecture based on OpenAI GPT
        which just uses the Decoder module of the Transformer architecture without
        any Encoder.
    """
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        self.N = N
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)

class DecoderHead(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class PhonemeTransformerOpenAIGPT(nn.Module):
    def __init__(self, model_path_or_name, phoneme_vocab, trg_vocab, d_phoneme, d_combined, N_phoneme,
            N_combined, heads_phoneme, heads_combined, dropout):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(model_path_or_name)
        self.phoneme_encoder = Encoder(phoneme_set_size, d_phoneme, N_phoneme, heads_phoneme, dropout)
        self.combined_decoder = DecoderHead(d_combined, N_combined, heads_combined, dropout)
        self.out = nn.Linear(d_combined, trg_vocab)

    def forward(self, src, phn, src_mask, phn_mask):
        e_masked = mask_context(src, src_mask)
        e_outputs = self.backbone(e_masked)
        phoneme_e_outputs = self.phoneme_encoder(phn, phn_mask)
        combined_outputs = torch.cat([e_outputs, phoneme_e_outputs], dim=1)
        outputs = self.combined_decoder(combined_outputs)

class PhonemeTransformer(nn.Module):
    def __init__(self, vocab_size, phoneme_vocab_size, d_trg, d_phoneme, d_combined, N_trg, N_phoneme,
            N_combined, heads_trg, heads_phoneme, heads_combined, dropout, max_length):
        super().__init__()
        self.max_length = max_length
        self.context_encoder = Encoder(vocab_size, d_trg, N_trg, heads_trg, dropout, max_length)
        self.phoneme_encoder = Encoder(phoneme_vocab_size, d_phoneme, N_phoneme, heads_phoneme, dropout, max_length)
        self.combined_decoder = DecoderHead(d_combined, N_combined, heads_combined, dropout)
        
        # Adaptive Softmax include fc layers with size dependent on bucket
        self.adaptivesoftmax = nn.AdaptiveLogSoftmaxWithLoss(
            d_combined, vocab_size, cutoffs=[
                # First bucket is special tokens (unk, pad, sos, eos, \n)
                5, 50, 500, 5000,
            ], div_value=2.0,
        )

    def forward(self, src, phn, src_mask, phn_lengths, targets, src_pad_mask): #phn_mask):
        B, T = src.shape
        _, _, P = phn.shape  
        assert T <= self.max_length
        assert phn.shape == (B, T, P)
        assert src_mask.shape == (B, T, T)
        assert targets.min().item() >= 0

        # Get word sequence embeddings.
        e_outputs = self.context_encoder(src, src_mask)

        # Get transformer hidden state representation of the phoneme
        # sequence for each word.
        phn_e_outputs = []
        for t in range(src.shape[1]):
                t_phn = phn[:,t,:] # Gives tensor of shape (B x MaxPhonemes)
                t_phn_e_outputs = self.phoneme_encoder(t_phn, None)
                t_phn_ids = phn_lengths[:,t].unsqueeze(1) - 1
                t_phn_ids = t_phn_ids.clamp(min=0)
                t_phn_ids = t_phn_ids.expand(t_phn_ids.shape[0], t_phn_e_outputs.shape[2]).unsqueeze(1)
                t_phn_e_output = t_phn_e_outputs.gather(1, t_phn_ids)
                phn_e_outputs.append(t_phn_e_output)

        # Combine phoneme encodings from word batchs for each timestep.
        phn_e_outputs = torch.cat(phn_e_outputs, dim=1)

        # Combine output from word context transformer and 
        combined_outputs = torch.cat([e_outputs, phn_e_outputs], dim=2)
        outputs = self.combined_decoder(combined_outputs, src_mask)


        # Have to mask the targets and outputs to get rid of padding
        # locations.
        outputs = outputs[src_pad_mask.byte()]
        targets = targets[src_pad_mask.byte()]

        # Apply adaptive softmax to the batch of predicted next words.
        outputs, loss = self.adaptivesoftmax(outputs, targets)

        # Get predictions based on most probable word for each time step and
        # use to determine batch accuracy.
        preds = outputs
        
        return outputs, loss

    def gen_word(self, src, phn, src_mask):
        pass

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-src_pad', default=0)
    parser.add_argument('-trg_pad', default=0)
    parser.add_argument('-device', default=0)
    opt = parser.parse_args()
    vocab_size = 10000
    phoneme_vocab_size = 100
    d_trg = 256
    d_phoneme = 256
    d_combined = d_trg + d_phoneme
    N_trg = 4
    N_phoneme = 3
    N_combined = 2
    heads_trg = 8
    heads_phoneme = 4
    heads_combined = 4
    dropout = 0.1
    max_length = 256
    model = PhonemeTransformer(
            vocab_size=vocab_size,
            phoneme_vocab_size=phoneme_vocab_size,
            d_trg=d_trg,
            d_phoneme=d_phoneme,
            d_combined=d_combined,
            N_trg=N_trg,
            N_phoneme=N_phoneme,
            N_combined=N_combined,
            heads_trg=heads_trg,
            heads_phoneme=heads_phoneme,
            heads_combined=heads_combined,
            dropout=dropout,
            max_length=max_length
            )
    if torch.cuda.is_available():
        model.cuda()
    null_token = 0
    start_token = 1
    end_token = 2
    vocab_low_token = 3
    vocab_high_token = 10000
    phoneme_low_token = 3
    phoneme_high_token = 100
    torch.manual_seed(12345)
    src = torch.zeros((1, 20))
    trg = torch.zeros((1, 20))
    src[:,0] = start_token
    src[:,1:11] = torch.randint(vocab_low_token, vocab_high_token, (1, 10))
    src = src.long()
    trg[:,0:10] = src[:,1:11]
    trg[:,10] = end_token
    trg = trg.long()

    src = src.cuda() if torch.cuda.is_available() else src
    trg = trg.cuda() if torch.cuda.is_available() else trg
    phn = torch.zeros(1, 20, 15) # Assuming no word has more than 15 phonemes.
    phn[:,0, 0] = start_token
    phn_lengths = torch.randint(1,14, (phn.size(0), phn.size(1)))
    for i in range(phn.size(0)):
        for t in range(phn.size(1)):
            z = phn_lengths[i,t]
            phn[i,t,0] = start_token
            phn[i,t,1:z+1] = torch.randint(phoneme_low_token, phoneme_high_token, (1, 1, z))
            phn[i,t,z] = end_token
    phn = phn.long()
    phn = phn.cuda() if torch.cuda.is_available() else phn
    phn_lengths = phn_lengths.cuda()
    _, src_mask = create_masks(src, src, opt)
    outputs = model(src, phn, src_mask, phn_lengths, src)
    import pdb; pdb.set_trace()
if __name__ == '__main__':
    main()
