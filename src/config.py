import torch

model_params = {
    'embed_size': 100,
    'hidden_size': 1024,
    'phoneme_embed_size': 20,
    'phoneme_hidden_size': 256,

    # fill these in w/ train vocab
    #'vocab_size': len(txt_vocab),
    #'phoneme_vocab_size': len(phoneme_vocab),
}

transformer_model_params = {
    'd_trg': 100,
    'd_phoneme': 20,
    'd_combined': 120,
    'N_trg': 4,
    'N_phoneme': 3,
    'N_combined': 2,
    'heads_trg': 5,
    'heads_phoneme': 5,
    'heads_combined': 5,
    'dropout': 0.1,
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

def get_transformer_model_params(txt_vocab, phoneme_vocab):
    params = transformer_model_params.copy()

    params['vocab_size'] = len(txt_vocab)
    params['phoneme_vocab_size'] = len(phoneme_vocab)

    return params

def load_model(ModelType, state_dict_path, device, **kwargs):
    model = ModelType(**kwargs).to(device)
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)

    return model


