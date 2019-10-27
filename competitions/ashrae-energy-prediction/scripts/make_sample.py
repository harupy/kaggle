import pandas as pd


def main():
    train = pd.read_pickle('input/train.pkl')
    test = pd.read_pickle('input/test.pkl')

    train.sample(n=100000, random_state=0).to_pickle('input/train_sample.pkl')
    test.sample(n=100000, random_state=0).to_pickle('input/test_sample.pkl')


if __name__ == '__main__':
    main()
