from utils import *
from cnn import CNN
from spark_cnn import SparkCNN
from time import time

cnn = CNN(1)
cnn.init_layers(10)

start = time()
X, Y = load_training_data(0, 2000)
RS = cnn.forward(X)
R1, R2, R3, R4 = RS
end = time()
print('naive cnn time cost %.3f' % (end - start))

spark_cnn = SparkCNN(1, 50)
spark_cnn.conv = cnn.conv
spark_cnn.relu = cnn.relu
spark_cnn.pool = cnn.pool
spark_cnn.fc = cnn.fc
R4_spark, Y_spark = spark_cnn.train(2000)


print('---')
print(np.argmax(R4, 1))
print('---')
print(np.argmax(R4_spark, 1))

assert np.allclose(R4, R4_spark), 'failed 1'
assert np.allclose(Y, Y_spark), 'failed 2'
