from __future__ import print_function
import os
import pickle
import random
import sys
import time
from keras.callbacks import Callback, EarlyStopping
from keras.layers import Dense, Dropout, Embedding, LSTM, TimeDistributed, GRU, RNN
from keras.optimizers import Adam
from keras.models import load_model, Sequential
from keras import backend as K
import numpy as np
from lyrics_vectorizer import Lyrics_Vectorizer
from utilities import cyan_color, green_color, red_color
from utilities import prediction_sample, stateful_RNN_shape, seed_search
from six.moves import cPickle

class ModelSampler(Callback):
    def __init__(self, meta_model):
        super(ModelSampler, self).__init__()
        self.meta_model = meta_model

    def on_epoch_end(self, epoch, logs=None):
        print()
        green_color('Sampling model...')
        self.meta_model.update_sample_model_weights()
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('Using diversity:', diversity)
            self.meta_model.sample(diversity=diversity)
            print('Saving Epoch Model!')
            self.meta_model.sample_model.save('builtmodel_e.h5')
            pickle.dump(self.meta_model, open('builtmodel_e.pkl', 'wb'))
            print('-' * 50)

class MetaModel:
    def __init__(self):
        self.train_model = None
        self.sample_model = None
        self.seeds = None
        self.vectorizer = None

    def _load_data(self, data_directory, word_level_flag, preserve_input, preserve_output,
                   batch_size, seq_length, seq_step):
        try:
            with open(os.path.join(data_directory, 'nslyrics.txt'), encoding="utf8") as lyrics_file:
                print(lyrics_file)
                text = lyrics_file.read()
        except FileNotFoundError:
            red_color("No lyrics.txt in data_directory")
            sys.exit(1)

        skip_validate = True
        try:
            with open(os.path.join(data_directory, 'validate_lyrics.txt')) as validate_file:
                lyrics_val = validate_file.read()
                skip_validate = False
        except FileNotFoundError:
            pass

        self.seeds = seed_search(text)
        all_text = text if skip_validate else '\n'.join([text, lyrics_val])
        self.vectorizer = Lyrics_Vectorizer(all_text, word_level_flag,
                                            preserve_input, preserve_output)

        data = self.vectorizer.vectorize(text)
        x, y = stateful_RNN_shape(data, batch_size, seq_length, seq_step)
        print('x.shape:', x.shape)
        print('y.shape:', y.shape)

        if skip_validate:
            return x, y, None, None

        data_val = self.vectorizer.vectorize(lyrics_val)
        x_val, y_val = stateful_RNN_shape(data_val, batch_size,
                                          seq_length, seq_step)
        print('x_val.shape:', x_val.shape)
        print('y_val.shape:', y_val.shape)
        return x, y, x_val, y_val

    def _build_models(self, batch_size, embedding_size, rnn_size, num_layers):
        model = Sequential()
        model.add(Embedding(self.vectorizer.vocab_size,
                            embedding_size,
                            batch_input_shape=(batch_size, None)))
        for layer in range(num_layers):
            """Manually change the RNN model here. TODO: Dynamically change from RNN to LSTM to GRU"""
            model.add(GRU(rnn_size,
                           stateful=True,
                           return_sequences=True))
        model.add(TimeDistributed(Dense(self.vectorizer.vocab_size,
                                        activation='softmax')))
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(lr=0.0001),
                      metrics=['accuracy', perplexity])
        model.summary()

        self.train_model = model
        config = model.get_config()
        #print(len(config['layers']))
        config[0]['config']['batch_input_shape'] = (1, None)
        self.sample_model = Sequential.from_config(config)
        self.sample_model.trainable = False

    def update_sample_model_weights(self):
        self.sample_model.set_weights(self.train_model.get_weights())

    def train(self, data_directory, word_level_flag, preserve_input, preserve_output,
              batch_size, seq_length, seq_step, embedding_size, rnn_size,
              num_layers, num_epochs, live_sample):
        green_color('Loading the data.....')
        load_start = time.time()
        x, y, x_val, y_val = self._load_data(data_directory, word_level_flag,
                                             preserve_input, preserve_output,
                                             batch_size, seq_length, seq_step)
        load_end = time.time()
        red_color('Data load time:', load_end - load_start)

        green_color('Building the model.....')
        model_start = time.time()
        self._build_models(batch_size, embedding_size, rnn_size, num_layers)
        model_end = time.time()
        red_color('Model build time', model_end - model_start)

        green_color('Training the model.....')
        train_start = time.time()
        early_stopping = EarlyStopping(monitor='loss', patience=3)
        validation_data = (x_val, y_val) if (x_val is not None) else None
        if live_sample:
            callbacks = [ModelSampler(self), early_stopping]
        else:
            callbacks = [early_stopping]

        history = self.train_model.fit(x, y,
                             validation_data=validation_data,
                             batch_size=batch_size,
                             shuffle=False,
                             epochs=num_epochs,
                             verbose=1,
                             callbacks=callbacks)
        self.update_sample_model_weights()
        train_end = time.time()
        red_color('Time of training', train_end - train_start)
        with open(os.path.join(data_directory, 'train_history'), 'wb') as file_pi:
            cPickle.dump(history.history, file_pi)

    def sample(self, seed=None, length=None, diversity=1.0):
        self.sample_model.reset_states()

        if length is None:
            length = 100 if self.vectorizer.word_level_flag else 500

        if seed is None:
            seed = random.choice(self.seeds)
            print('Seed used: ', end='')
            cyan_color(seed)
            print('-' * 50)

        preds = None
        seed_vector = self.vectorizer.vectorize(seed)
        cyan_color(seed, end=' ' if self.vectorizer.word_level_flag else '')
        for char_index in np.nditer(seed_vector):
            preds = self.sample_model.predict(np.array([[char_index]]),
                                              verbose=0)

        sampled_indices = np.array([], dtype=np.int32)
        for i in range(length):
            char_index = 0
            if preds is not None:
                char_index = prediction_sample(preds[0][0], diversity)
            sampled_indices = np.append(sampled_indices, char_index)
            preds = self.sample_model.predict(np.array([[char_index]]),
                                              verbose=0)
        sample = self.vectorizer.unvectorize(sampled_indices)
        print(sample)
        return sample

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['train_model']
        del state['sample_model']
        return state

def save(model, data_directory):
    print('saving model!')
    model.sample_model.save('builtmodel.h5')
    pickle.dump(model, open('builtmodel.pkl', 'wb'))

def load(data_directory):
    keras_file_path = os.path.join('builtmodel_e.h5')
    pickle_file_path = os.path.join('builtmodel_e.pkl')
    model = pickle.load(open(pickle_file_path, 'rb'))
    model.sample_model = load_model(keras_file_path)
    return model

"""Text generation should be measured by Perplexity, although cross-entropy should be okay too"""
def perplexity(y_true, y_pred):
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    perplexity = K.pow(2.0, cross_entropy)
    return perplexity
