import pickle

from util import ROOT_DIR


def unpickle(f_name):
    with open(f_name, 'rb') as f:
        return pickle.load(f, encoding='bytes')


def get_data():
    train_data = {'data': [], 'labels': []}
    for i in range(1, 6):
        batch_data = unpickle(f'{ROOT_DIR}/cifar/data_batch_{i}')
        train_data['data'].extend(batch_data[b'data'])
        train_data['labels'].extend(batch_data[b'labels'])
    test_batch_data = unpickle(f'{ROOT_DIR}/cifar/test_batch')
    test_data = {'data': test_batch_data[b'data'], 'labels': test_batch_data[b'labels']}
    return train_data, test_data
