import gc
import pandas as pd

from preprocess import reduce_mem_usage


def main():
    # train data
    train = pd.read_csv('input/train.csv', parse_dates=['timestamp'])
    weather_train = pd.read_csv('input/weather_train.csv', parse_dates=['timestamp'])
    building = pd.read_csv('input/building_metadata.csv')

    train = train.merge(building, left_on='building_id', right_on='building_id', how='left')
    train = train.merge(weather_train, left_on=['site_id', 'timestamp'], right_on=['site_id', 'timestamp'])
    del weather_train
    gc.collect()

    reduce_mem_usage(train)
    train.to_pickle('input/train.pkl')
    print('train done')
    del train
    gc.collect()

    # test data
    test = pd.read_csv('input/test.csv', parse_dates=['timestamp'])
    weather_test = pd.read_csv('input/weather_test.csv', parse_dates=['timestamp'])

    test = test.merge(building, left_on='building_id', right_on='building_id', how='left')
    test = test.merge(weather_test, left_on=['site_id', 'timestamp'], right_on=['site_id', 'timestamp'], how='left')
    del weather_test, building
    gc.collect()

    reduce_mem_usage(test)
    test.to_pickle('input/test.pkl')
    print('test done')


if __name__ == '__main__':
    main()
