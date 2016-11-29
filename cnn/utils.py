import os
import numpy as np
import pickle

def load_data():
    print('Loading cifar10 data...')
    curpath = os.path.dirname(os.path.realpath(__file__))
    dirpath = os.path.join(curpath, os.path.pardir, 'cifar10')

    print('Reading classifications...')
    classifications = None
    with open(os.path.join(dirpath, "batches.meta"), 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
        classifications = raw['label_names']

    print('Reading training set...')
    X_train = []
    Y_train = []
    for i in range(1, 2):
        filename = 'data_batch_' + str(i)
        with open(os.path.join(dirpath, filename), 'rb') as f:
            raw = pickle.load(f, encoding='latin1')
            data = raw['data'].reshape(10000, 3, 32, 32).astype("float").transpose(0, 3, 2, 1)
            X_train.append(data)
            Y_train.append(raw['labels'])
    X_train = np.concatenate(X_train)
    Y_train = np.concatenate(Y_train)

    print('Reading testing set...')
    X_test = None
    Y_test = None
    with open(os.path.join(dirpath, filename), 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
        X_test = raw['data'].reshape(10000, 3, 32, 32).transpose(0, 3, 2, 1)
        Y_test = np.array(raw['labels'])

    print('Data loading done')
    return classifications, X_train, Y_train, X_test, Y_test

# softmax loss calculation
def softmax(S, Y):
    # input: S is scores of images [N x C] C is number of classifications
    # input: Y is correct labels for images [N], each entry is in classifications
    # output: L softmax loss and dS gradients of L on S
    N, C = S.shape

    # calculate probabilities [N x C]
    ps = np.exp(S - np.max(S, 1, None, True))
    ps = ps / np.sum(ps, 1, None, None, True)

    # calculate loss
    L = np.sum(0.0 - np.log(ps[np.arange(N), Y])) / N

    # calculate gradients
    dS = ps.copy()
    dS[np.arange(N), Y] -= 1
    dS = dS / N

    return L, dS
