from config import *

def read_data(files):
    class cifar10(object):
        pass

    result = cifar10()
    result.label_size = LABEL_SIZE
    result.image_height = IMAGE_HEIGHT
    result.image_width = IMAGE_WIDTH
    result.image_depth = IMAGE_DEPTH
    result.image_size = result.image_height * result.image_width * result.image_depth
    result.entry_size = result.label_size + result.image_size
    
    reader = tf.FixedLengthRecordReader(record_bytes = result.entry_size)
    result.key, value = reader.read(files)
    entry = tf.decode_raw(value, tf.uint8)
    result.label = tf.cast(tf.slice(entry, [0], [result.label_size]), tf.int32)

    result.image = tf.reshape(tf.slice(entry, [result.label_size], [result.image_size]), [result.image_depth, result.image_height, result.image_width])

    result.image = tf.transpose(result.image, [1, 2, 0])
    result.image = tf.cast(result.image, tf.float32)

    return result

def generate_batch(image, label, batch_size, min_queue_size):
    thread_num = 16
    image, label = tf.train.shuffle_batch([image, label], batch_size = batch_size, num_threads = thread_num, capacity = min_queue_size + 3 * batch_size, min_after_dequeue = min_queue_size)
    return image, tf.one_hot(tf.reshape(label, [batch_size]), depth = LABEL_NUM)

def train_input(batch_size):
    files = glob.glob(os.path.join(CIFAR10_PATH, "data_batch_*.bin"))
    files = tf.train.string_input_producer(files)
    input = read_data(files)
    min_queue_size = np.int32(FRAC_MIN_QUEUE_SIZE * TRAIN_EPOCH_NUM) 
    return generate_batch(input.image, input.label, batch_size, min_queue_size)

def test_input(batch_size):
    files = glob.glob(os.path.join(CIFAR10_PATH, "test_batch.bin"))
    files = tf.train.string_input_producer(files)
    input = read_data(files)
    min_queue_size = np.int32(FRAC_MIN_QUEUE_SIZE * TRAIN_EPOCH_NUM) 
    return generate_batch(input.image, input.label, batch_size, min_queue_size)

