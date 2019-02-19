Deep CNN practice for MNIST data set

Using TensorFlow and Keras respectively

CNN structure:

Conv Layer #1, filters 32, kernel size 3, strides 1
max pooling layer #1, kernel size 2, strides 2

Conv Layer #2, filters 64, kernel size 3, strides 1
Max pooling layer #2, kernel size 2, strides 2

Conv Layer #3, filters 128, kernel size 3, strides 1
Max pooling layer #3, kernel size 2, strides 2

Full connection Layer #4, units 625

Output layer #5, units 10

Training dataset accuracy: 0.9772
Testing dataset accuracy: 0.9761
