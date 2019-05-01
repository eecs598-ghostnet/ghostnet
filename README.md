# ghostnet
## Dependencies



```
pip install -r src/requirements.txt
```

   
## 1. Data preprocessing

We use the [Spotipy API](https://github.com/plamere/spotipy), [Spotify Web API](https://developer.spotify.com/documentation/web-api/), [Genius API](https://genius.com/developers), and the [Lyric Genius API](https://github.com/johnwmillr/LyricsGenius)

In this section, we: 
- generating the discography, then lyrics, of artists 
- preprocessing the data to remove choruses, and nonlyrical information
- adding phoneme information of the data

Run from src directory:
```
saveArtist.py
compilelyrics.py
test.py
```

This gives you a Corpus instance with the artist's lyrics specified in test.
Details of the Corpus class can be seen in data.py

Note: espeak-ng should be installed on the machine. It is available here: https://github.com/espeak-ng/espeak-ng
Code may also work with espeak with slight modifications to english2phonemes.py, but it has not been tested.

## 2. GhostNet-LSTM
### Training
Run from the src directory: 
```
python3 train.py [model_weight_checkpoint_path]
```

### Generating new samples
Run from the src directory: 
```
python3 gen_text.py ["seed phrase"]
```
If no seed phrase is given, 100 leading lines from the test dataset will be used as seed phrases.


## 3. GhostNet-Transformer
We incorporated [GPT2 for Pytorch](https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/modeling_gpt2.py)


### Training  
Run from the src directory: 
```
python3 train_transformer.py
```

### Generate new samples
Run from the src directory: 
```
python3 gen_text_transformer.py
```

## 4. Baselines
### Backus-Naur 
We modified an implementation found at: https://github.com/schollz/poetry-generator


### Deep-Rhyme
We modified an implementation found at: https://github.com/mikesj-public/deep-rhyme



### Deep-Rapping
We modified an implementation given by the authors of: https://www.ijitee.org/wp-content/uploads/papers/v8i2s/BS2651128218.pdf
To sample, from baselines-evaluators/DeepRapping/: 
```
python sample_the_model.py --diversity == 0.5
```


### LSTM_Markov
Code adapted from: https://www.kaggle.com/paultimothymooney/poetry-generator-rnn-markov/notebook
To train or sample, from baselines-evaluators/PoetryGen-RNNMark/:
```
python kaggleRNN.py
```
Hyperparameters and modes can be changed within this script as well. 



## 5. Metrics
### Similarity 
From baselines-evalators/bdata/
```
calc_sim.py [textfile]
```
### Perplexity 
From baselines-evalators/bdata/
```
calc_perp.py [textfile]
```
### BLEU
From baselines-evaulators/bdata/
```
calc_BLEU.py [text file]
```

### Rhyme Density
Downloaded from: https://sourceforge.net/projects/rhymeanalyzer/
From baselines-evaluators/RAdist/
```
java -jar "RhymeApp.jar" 
```
Paste the poetry text into its text box, run teh application. Rhyme Density is listed as a metric.  



