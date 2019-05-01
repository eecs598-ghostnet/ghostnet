# ghostnet
## Dependencies

## Overview
This repo includes the source files our project used for
1) - generating the discography, then lyrics, of artists 
   - preprocessing the data to remove choruses, and nonlyrical information
   - adding phoneme information of the data
2) - training and testing our LSTM model with phoneme embeddings
3) - training and testing our Transformer model with phoneme embeddings
4) - comparison baselines we used
5) - Evaluating a .txt file of lyrics with our provided metrics
   
## 1. Data preprocessing
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

## 3. GhostNet-Transformer

## 4. Baselines

## 5. Metrics




