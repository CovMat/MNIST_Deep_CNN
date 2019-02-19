from read_file import read_file
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout

# Define some parameters and hyperparameters
nb_classes = 10
my_learning_rate = 0.0003
logdir = "./logs/test1"
# conv layer 1
nb_filters_conv1 = 32
size_filters_conv1 = 3
size_pool1 = 2
stride_pool1 = 2
dropout_rate1 = 0.3
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
X_train = np.asarray( train_image_data ).reshape( train_image_num, row_num, col_num, 1  )
Y_train = np.zeros( (train_label_num, nb_classes) )
for i in range(nb_classes):
    Y_train[ : , i ] = ( np.asarray(train_label_data) == i )
X_test = np.asarray( test_image_data ).reshape( test_image_num, row_num, col_num, 1  )
Y_test = np.zeros( (test_label_num, nb_classes) )
for i in range(nb_classes):
    Y_test[ : , i ] = ( np.asarray(test_label_data) == i )


# Build the CNN
model = Sequential()
# Conv Layer #1
model.add(\
           Conv2D( filters = nb_filters_conv1,\
                   kernel_size = size_filters_conv1,\
                   padding = 'same',\
                   activation = 'relu',\
                   input_shape = (row_num, col_num, 1)  )\
          )
model.add(\
           MaxPooling2D( pool_size = size_pool1, strides = stride_pool1, padding = 'SAME'  )\
          )
model.add( Dropout( rate = dropout_rate1 ) )
# Conv Layer #2
model.add(\
           Conv2D( filters = nb_filters_conv2,\
                   kernel_size = size_filters_conv2,\
                   padding = 'same',\
                   activation = 'relu' )\
          )
model.add(\
           MaxPooling2D( pool_size = size_pool2, strides = stride_pool2, padding = 'SAME'  )\
          )
model.add( Dropout( rate = dropout_rate2 ) )
# Conv Layer #3
model.add(\
           Conv2D( filters = nb_filters_conv3,\
                   kernel_size = size_filters_conv3,\
                   padding = 'same',\
                   activation = 'relu' )\
          )
model.add(\
           MaxPooling2D( pool_size = size_pool3, strides = stride_pool3, padding = 'SAME'  )\
          )
model.add( Dropout( rate = dropout_rate3 ) )
# FC layers #4
model.add( Flatten() )
model.add( Dense( units = nb_units4, activation = 'relu' ) )
model.add( Dropout( rate = dropout_rate4 ) )
# FC layers #5
model.add( Dense( units = nb_units5, activation = 'relu' ) )
# 
model.compile( optimizer = 'Adam', loss = 'binary_crossentropy' )
model.fit( x = X_train, y = Y_train, batch_size = 512, epochs = 20 )
score = model.evaluate( x = X_test, y = Y_test )
