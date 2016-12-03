import os
import psutil
import numpy as np
import pickle

curpath = os.path.dirname(os.path.realpath(__file__))
dirpath = os.path.join(curpath, os.path.pardir, 'cifar10')
parpath = os.path.join(curpath, 'parameters')

def save_parameters(name, data):
    with open(os.path.join(parpath, name), 'wb') as f:
        pickle.dump(data, f)

def load_parameters(name):
    data = None
    with open(os.path.join(parpath, name), 'rb') as f:
        data = pickle.load(f)
    return data

def load_classifications():
    print('Loading classifications...')
    classifications = None
    with open(os.path.join(dirpath, "batches.meta"), 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
        classifications = raw['label_names']
    return classifications

def load_testing_data(start = 0, end = 10000):
    assert end > start and start >= 0 and end <= 10000, 'invalid test data range'
    print('Loading testing set...')
    filename = 'test_batch'
    X = None
    Y = None
    with open(os.path.join(dirpath, filename), 'rb') as f:
        raw = pickle.load(f, encoding='latin1')
        X = raw['data'].reshape(10000, 3, 32, 32).transpose(0, 3, 2, 1)
        Y = np.array(raw['labels'])

    # make RGB in the range of [-0.5, 0.5]
    X = X / 255.0 - 0.5

    return X[start:end, :, :, :], Y[start:end]

def load_training_data(start = 0, end = 60000):
    assert end > start, 'invalid range'
    # load data from disc in the range [start, end)
    print('Reading training set...')
    print('Loading cifar10 data from ' + str(start) + ' to ' + str(end - 1))
    file_start = start // 10000 + 1
    file_end = (end - 1) // 10000 + 1

    X = []
    Y = []

    for i in range(file_start, file_end + 1):
        filename = 'data_batch_' + str(i)
        with open(os.path.join(dirpath, filename), 'rb') as f:
            raw = pickle.load(f, encoding='latin1')
            data = raw['data'].reshape(10000, 3, 32, 32).astype('float').transpose(0, 3, 2, 1)
            labels = raw['labels']
            if i == file_start and i == file_end:
                # load part of the file
                start_pos = start % 10000
                end_pos = end % 10000
                if (end_pos == 0):
                    end_pos = 10000
                X.append(data[start_pos:end_pos, :, :, :])
                Y.append(labels[start_pos:end_pos])
            elif i == file_start:
                # load part of the file
                start_pos = start % 10000
                X.append(data[start_pos:, :, :, :])
                Y.append(labels[start_pos:])
            elif i == file_end:
                # load part of the file
                end_pos = end % 10000
                if (end_pos == 0):
                    end_pos = 10000
                X.append(data[:end_pos, :, :, :])
                Y.append(labels[:end_pos])
            else:
                # load the entire file
                X.append(data)
                Y.append(labels)

    X = np.concatenate(X)
    Y = np.concatenate(Y)

    # make RGB in the range of [-0.5, 0.5]
    X = X / 255.0 - 0.5

    return X, Y

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

# helper for profiling memory usage
def memory():
    process = psutil.Process(os.getpid())
    return str(process.memory_info().rss // 1024 // 1024) + "MB"
