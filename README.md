# ghostnet
## Data preprocessing
Run from src directory
1. saveArtist.py
2. compilelyrics.py
3. test.py

This gives you a Corpus instance with the artist's lyrics specified in test.
Details of the Corpus class can be seen in data.py

Note: espeak-ng should be installed on the machine. It is available here: https://github.com/espeak-ng/espeak-ng
Code may also work with espeak with slight modifications to english2phonemes.py, but it has not been tested.

## Install dependencies
```
pip install -r src/requirements.txt
```

## Training Models

To train the transformer based model, run the following commands from within the src directory:
```
python3 train_transformer.py
```

## Generate Examples

To generate new samples, run the following command from within the src directory:
```
python3 gen_text_transformer.py
```
