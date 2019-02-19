from read_file import read_file
import numpy as np
import tensorflow as tf
from tensorflow.layers import conv2d, max_pooling2d, dropout, flatten, dense

# Define some parameters and hyperparameters
nb_classes = 10
my_learning_rate = 0.003
logdir = "./logs/test1_droprate_0_learning_rate_0.003"
# conv layer 1
nb_filters_conv1 = 32
size_filters_conv1 = 3
size_pool1 = 2
stride_pool1 = 2
dropout_rate1 = 0
# conv layer 2
nb_filters_conv2 = 64
size_filters_conv2 = size_filters_conv1
size_pool2 = size_pool1
stride_pool2 = stride_pool1
dropout_rate2 = dropout_rate1
# conv layer 3
nb_filters_conv3 = 128
size_filters_conv3 = size_filters_conv1
size_pool3 = size_pool1
stride_pool3 = stride_pool1
dropout_rate3 = dropout_rate1
# FC layer 4
nb_units4 = 625
dropout_rate4 = dropout_rate1
# FC layer 5
nb_units5 = nb_classes

# read the training data set:
train_label_num, _, _, train_label_data = read_file( "data/train-labels-idx1-ubyte" )
train_image_num, row_num, col_num, train_image_data = read_file( "data/train-images-idx3-ubyte" )
# read the testing data set
test_label_num, _, _, test_label_data = read_file( "data/t10k-labels-idx1-ubyte" )
test_image_num, _, _, test_image_data = read_file( "data/t10k-images-idx3-ubyte" )
# transfer label and image data to proper format
X_train = np.asarray( train_image_data ).reshape( train_image_num, row_num*col_num  )
X_train -= 128
Y_train = np.asarray( train_label_data )
X_test = np.asarray( test_image_data ).reshape( test_image_num, row_num*col_num )    
X_test -= 128

# Build the CNN
X_input = tf.placeholder( tf.float32, [None, row_num*col_num] )
Y_input = tf.placeholder( tf.int32, [None] )
X_img = tf.reshape( X_input, [-1, row_num, col_num, 1] )
is_training = tf.placeholder( tf.bool )
# conv layer #1
conv1 = conv2d( X_img, nb_filters_conv1, size_filters_conv1, 1, 'same',\
                activation=tf.nn.relu )
pool1 = max_pooling2d( conv1, size_pool1, stride_pool1, 'same' )
dpt1 = dropout( pool1, dropout_rate1, training = is_training )
# conv layer #2
conv2 = conv2d( dpt1, nb_filters_conv2, size_filters_conv2, 1, 'same',\
                activation=tf.nn.relu )
pool2 = max_pooling2d( conv2, size_pool2, stride_pool2, 'same' )
dpt2 = dropout( pool2, dropout_rate2, training = is_training )
# conv layer #3
conv3 = conv2d( dpt2, nb_filters_conv3, size_filters_conv3, 1, 'same',\
                activation=tf.nn.relu )
pool3 = max_pooling2d( conv3, size_pool3, stride_pool3, 'same' )
dpt3 = dropout( pool3, dropout_rate3, training = is_training )
# FC layer #4
FC4 = flatten( dpt3 )
FC4 = dense( FC4, nb_units4, tf.nn.relu )
dpt4 = dropout( FC4, dropout_rate4, training = is_training )
# FC layer #5
FC5 = dense( dpt4, nb_units5 )
cost = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( labels=Y_input, logits=FC5 ) )
optimizer = tf.train.AdamOptimizer( learning_rate = my_learning_rate ).minimize( cost )
# for TensorBoard
cost_summ = tf.summary.scalar( "Cost Function", cost )

# run (mini-batch)
summary = tf.summary.merge_all()
sess = tf.Session()
sess.run( tf.global_variables_initializer() )
writer = tf.summary.FileWriter( logdir )
writer.add_graph( sess.graph )
#
batch_size = 512
batch_num = train_label_num // batch_size
#
global_step = 0
for epoch in range(2):
    for i in range(batch_num):
        st = i*batch_size
        ed = (i+1)*batch_size
        if ( i == batch_num-1 ):
            ed = train_label_num
        s, _ = sess.run( [ summary, optimizer], feed_dict={\
                                        X_input:X_train[ st:ed, :],\
                                        Y_input:Y_train[ st:ed ],\
                                        is_training:True} )
        writer.add_summary( s, global_step = global_step )
        global_step += 1

# Accuracy
logit_train = sess.run( FC5, feed_dict={X_input:X_train, is_training:False} )
max_index = sess.run( tf.argmax(logit_train, axis=1) )
accuracy = sess.run( tf.reduce_mean(tf.cast( max_index == train_label_data, dtype=tf.float32) ) )
print( "Accuracy of Training set:"+ str(accuracy) )
logit_test = sess.run( FC5, feed_dict={X_input:X_test, is_training:False} )
max_index = sess.run( tf.argmax(logit_test, axis=1) )
accuracy = sess.run( tf.reduce_mean(tf.cast( max_index == test_label_data, dtype=tf.float32) ) )
print( "Accuracy of Testing set:"+ str(accuracy) )
