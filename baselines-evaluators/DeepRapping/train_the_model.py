from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from lyrics_gen_model import MetaModel, save

"""Train your lyrics model"""
def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-directory', type=str, default='data/lyrics') # Directory of training lyrics
    parser.add_argument('--live-sample', action='store_false')               # Flag: Sample lyrics during training
    parser.add_argument('--word-level-flag', action='store_true')            # Flag: True - Generate at word level; Else: Generate at character level
    parser.add_argument('--preserve-input', action='store_false')            # Preserve original type case of the input
    parser.add_argument('--preserve-output', action='store_true')            # Preserve original type case of the output
    parser.add_argument('--embedding-size', type=int, default=64)            # Size of embedding
    parser.add_argument('--rnn-size', type=int, default=256)                 # Nodes per layer
    parser.add_argument('--num-layers', type=int, default=1)                 # Number of hidden layers
    parser.add_argument('--batch-size', type=int, default=256)               # Batch size
    parser.add_argument('--seq-length', type=int, default=50)                # Length of the lyrics sequence during training
    parser.add_argument('--seq-step', type=int, default=25)                  # Specify how often to take a training sequence from the data
    parser.add_argument('--num-epochs', type=int, default=1)                 # Epochs to train
    args = parser.parse_args()

    model = MetaModel()
    model.train(**vars(args))
    save(model, args.data_directory)

if __name__ == '__main__':
    main()
