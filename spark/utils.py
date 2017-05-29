import os
import psutil
import numpy as np
import pickle
from hdfs import InsecureClient
from redis import StrictRedis as redis

# constants
dirpath = os.path.join('/Users/Vivi', 'data', 'cifar10')
perpath = os.path.join('/Users/Vivi', 'data', 'parameters')

redis_addresses = [
    ('127.0.0.1', 6379),
    ('himrod-5', 6379),
    ('himrod-6', 6379),
    ('himrod-7', 6379)
]

def get_hdfs_address():
    return 'http://himrod-5:50070'

def get_hdfs_address_spark():
    return 'hdfs://himrod-5'

def get_hdfs_client():
    return InsecureClient(get_hdfs_address(), root='/')

def save_parameters_local(name, data):
    name = os.path.join(perpath, name + '.params')
    with open(name, 'wb') as f:
        pickle.dump(data, f)

def load_parameters_local(name):
    name = os.path.join(perpath, name + '.params')
    data = None
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def save_parameters(name, data):
    name = 'parameters/' + name + '.params'
    client = get_hdfs_client()
    with client.write(name, overwrite=True) as writer:
        pickle.dump(data, writer)

def load_parameters(name):
    # use hdfs
    name = 'parameters/' + name + '.params'
    data = None
    client = get_hdfs_client()
    with client.read(name) as reader:
        data = pickle.load(reader)
    return data

def save_matrix(name, data):
    name = 'matrices/' + name + '.matrix'
    client = get_hdfs_client()
    with client.write(name, overwrite=True) as writer:
        pickle.dump(data, writer)

def load_matrix(name):
    name = 'matrices/' + name + '.matrix'
    data = None
    client = get_hdfs_client()
    with client.read(name) as reader:
        data = pickle.load(reader)
    return data

def save_matrix_redis(name, data):
    client = redis(host='127.0.0.1', port=6379, db=0)
    name = str(name)
    dtype = str(data.dtype)
    shape = str(data.shape)
    key = '{0}|{1}|{2}'.format(name, dtype, shape)
    client.set(key, data.ravel().tostring())
    return key

def load_matrix_redis(key):
    data = None
    for server in redis_addresses:
        host = server[0]
        port = server[1]
        client = redis(host=host, port=port, db=0)
        try:
            entry = client.get(key)
            if entry != None:
                dtype_str = key.split('|')[1]
                shape_str = key.split('|')[2]
                shape = []
                for s in shape_str[1:-1].split(','):
                    shape.append(int(s))
                data = np.fromstring(entry, dtype=dtype_str).reshape(tuple(shape))
                break
        except Exception as error:
            continue
    return data

def clear_matrix_redis():
    for server in redis_addresses:
        host = server[0]
        port = server[1]
        client = redis(host=host, port=port, db=0)
        try:
            client.flushdb()
        except Exception as error:
            continue

def save_batch(batch):
    name = 'batches/' + str(batch) + '.batch'
    client = get_hdfs_client()
    client.write(name, str(batch), overwrite=True)

def clear_batches():
    client = get_hdfs_client()
    client.delete('batches', recursive=True)

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
