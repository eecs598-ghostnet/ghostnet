# ghostnet
## Dependencies



```
pip install -r src/requirements.txt
```

   
## 1. Data preprocessing
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
Dylan train
```

### Generating new samples
Run from the src directory: 
```
Dylan test
```


## 3. GhostNet-Transformer
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
### Deep-Rhyme
We trained and tested an implementation found at: 



### Deep-Rapping
We trained and tested an implementation found at:



### LSTM_Markov
We trained and tested an implementation found at:




## 5. Metrics
###

###

###

###



