# spark-cnn
CS848 Final Project (using spark to speed up CNN)

### Note
The spark folder contains the final code for the project.
The cnn folder was firstly used for the creation of the naive approach, and can now be ignored.

### File Structure
In side the spark folder, we have
- cnn.py the naive implementation, used as base class for Spark CNN
- spark_cnn.py Spark enabled CNN with HDFS
- locality_cnn.py Spark enabled CNN with Redis
- conv.py implementation of convolution layer
- fc.py implementation of fully connected layer
- pool.py implementation of pooling layer
- relu.py implementation of rectified linear unit layer
- matrix.pyx (matrix_c.c and matrix_c.h) cython (native C) implementation of im2col and col2im
- utils.py utility functions for data I/O
- setup.py for compiling matrix.pyx in place

In the root folder of this project (spark-cnn), we have
- setup.py for packaging our source code to use for Spark
- naive_train.py training naive CNN implementation
- spark_train.py training Spark CNN with HDFS
- locality_train.py training Spark CNN with Redis

We also have some files for testing, which will be explained later in the Running Code section.

### Dependency
The project depends on Python3 and Python3 modules psutil, cython, hdfs, redis, and pyspark. <br>
The naive implementation can be run locally.<br>
Spark implementation (both HDFS and Redis) must be run through spark-submit.<br>
The HDFS version must have access to HDFS cluster with proper read/write permission to hdfs/matrices.<br>
The Redis version must have access to Redis server hosted on all cluster nodes.

### Before Running Code
To run our code, one must
- install all dependency
- setup Spark, HDFS, Redis accordingly
- copy cifar10 dataset to proper location on the Spark driver machine
- change node addresses in spark/utils.py
- package source code to generate a Python egg
- submit entry file with the generated egg to Spark with proper arguments

### Cluster Setup
The code should work with latest Spark cluster, but make sure some parameters are set correctly in utils.py to ensure that you have the cifar10 training data set at Spark driver machine, also the folder to save parameters are created at the driver machine beforehand.

To setup HDFS cluster, for each node that is used as a Spark worker node, it must also be a HDFS datanode. You can use the Spark driver node as the HDFS namenode. For best performance, please set replication factor of your HDFS cluster to one. Make sure you only have one namenode. Remember to set the permission accordingly so your Spark process has permission to access the location hdfs/matrices for HDFS powered Spark CNN.

To setup Redis, please install Redis server to all Spark worker node. Make sure your Redis server is allowing remote access.

### Modifying utils.py
There are some settings at the beginning of the file you must make sure are set correctly
- dirpath location to the cifar10 data sets on Spark driver machine
- perpath location to the perameter storage folder on Spark driver machine
- redis_addresses a list of all nodes' Redis server bindings, make sure 127.0.0.1:6379(or your own port) is always the first because we always want to try to fetch data from local node first.
- get_hdfs_address() need to return the address to HDFS namenode's HTTP api
- get_hdfs_address_spark() need to return the address to HDFS namenode

### Source Code Packaging
When submitting job to Spark, we must have all of our source code first packaged as a Python egg, to do this please run the following commands in root folder of this project (spark-cnn).
1. python3 setup.py build
2. python3 setup.py sdist
3. python3 setup.py bdist_egg
This will generate the egg file under the dist folder.

### Running Code
Now we can finally run our code, for naive CNN, you can do the following at the root folder (spark-cnn) directly
- python3 naive_train.py [num images] [num iterations] ###training naive CNN
- python3 naive_test.py [num images] ###testing naive CNN against cifar10 testing data set
- python3 naive_training_test.py [num images] ###testing naive CNN against cifar10 training data set

For Spark CNN with HDFS you must submit job via Spark, please run with your spark-submit script the following
- spark-submit --master [master address] (spark settings) --py-files [path to your generated egg file] [path to spark_train.py] [num images] [num iterations] [num batches] ###training
- spark-submit --master [master address] (spark settings) --py-files [path to your generated egg file] [path to spark_test.py] [num images] [num iterations] [num batches] ###testing against testing data set
- spark-submit --master [master address] (spark settings) --py-files [path to your generated egg file] [path to spark_training_test.py] [num images] [num iterations] [num batches] ###testing against training data set

For Spark CNN with Redis
- spark-submit --master [master address] (spark settings) --py-files [path to your generated egg file] [path to locality_train.py] [num images] [num iterations] [num batches]
- spark-submit --master [master address] (spark settings) --py-files [path to your generated egg file] [path to locality_test.py] [num images] [num iterations] [num batches]
- spark-submit --master [master address] (spark settings) --py-files [path to your generated egg file] [path to locality_training_test.py] [num images] [num iterations] [num batches]

You may need to supply spark settings to increase memory allowance for your Spark driver and executor, e.g.:
--driver-memory 32g --executor-memory 64g

the number of batches supplied to Spark CNN is the number of batches we are splitting training input images to, as well as the tasks (one for one batch) we are creating for Spark.
