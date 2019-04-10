model_params = {
    'embed_size': 100,
    'hidden_size': 1024,
    'phoneme_embed_size': 20,
    'phoneme_hidden_size': 256,

    # fill these in w/ train vocab
    #'vocab_size': len(txt_vocab),
    #'phoneme_vocab_size': len(phoneme_vocab),
}


def get_ae_params(phoneme_vocab):
    sos_token = phoneme_vocab.stoi['<sos>']

    ae_params = model_params.copy()

    ae_params.pop('embed_size')
    ae_params.pop('hidden_size')
    ae_params['phoneme_vocab_size'] = len(phoneme_vocab)
    ae_params['sos_token'] = sos_token

    return ae_params


def get_model_params(txt_vocab, phoneme_vocab):
    params = model_params.copy()

    params['vocab_size'] = len(txt_vocab)
    params['phoneme_vocab_size'] = len(phoneme_vocab)

    return params
