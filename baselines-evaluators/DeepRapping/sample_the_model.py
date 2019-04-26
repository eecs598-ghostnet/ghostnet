from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from lyrics_gen_model import load

"""Generate lyrics using the built model"""
def main():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-directory', type=str, default='data/lyrics/') # Directory of training lyrics
    parser.add_argument('--seed', type=str, default=None)                    # Leave this as is..
    parser.add_argument('--length', type=int, default=600)                  # Specify how the length of lyrics to generate
    parser.add_argument('--diversity', type=float, default=1.0)              # Specify the diversity of lyrics to generate
    args = parser.parse_args()

    model = load(args.data_directory)
    del args.data_directory
    model.sample(**vars(args))

if __name__ == '__main__':
    main()
