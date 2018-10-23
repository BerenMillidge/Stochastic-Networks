# trying out the various necessary techniques in a basic tensorflow model before translation into the real and cmoplex one!



import tensorflow as tf 
import numpy as np

# try withthe debuger
from tensorflow.python import debug as tf_debug

print(tf.__version__)

#dataset = tf.keras.datasets.mnist.load_data()
#print(type(dataset))
#(xtrain, xtest), (ytrain, ytest)  = dataset
#print(type(xtrain))
#print(xtrain.shape)

img_h = img_w = 28             # MNIST images are 28x28
img_size_flat = img_h * img_w  # 28x28=784, the total number of pixels
n_classes = 10                 # Number of classes, one class per digit
 
def load_data(mode='train'):
    """
    Function to (download and) load the MNIST data
    :param mode: train or test
    :return: images and the corresponding labels
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    if mode == 'train':
        x_train, y_train, x_valid, y_valid = mnist.train.images, mnist.train.labels, \
                                             mnist.validation.images, mnist.validation.labels
        return x_train, y_train, x_valid, y_valid
    elif mode == 'test':
        x_test, y_test = mnist.test.images, mnist.test.labels
    return x_test, y_test

def randomize(x, y):
    """ Randomizes the order of data samples and their corresponding labels"""
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]
    return shuffled_x, shuffled_y

def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]
    return x_batch, y_batch
 
 # Load MNIST data
x_train, y_train, x_valid, y_valid = load_data(mode='train')
print("Size of:")
print("- Training-set:\t\t{}".format(len(y_train)))
print("- Validation-set:\t{}".format(len(y_valid)))



# Hyper-parameters
epochs = 10             # Total number of training epochs
batch_size = 100        # Training batch size
display_freq = 100      # Frequency of displaying the training results
learning_rate = 0.001   # The optimization initial learning rate
 
h1 = 200                # Number of units in the first hidden layer

# weight and bais wrappers
def weight_variable(name, shape):
    """
    Create a weight variable with appropriate initialization
    :param name: weight name
    :param shape: weight shape
    :return: initialized weight variable
    """
    initer = tf.truncated_normal_initializer(stddev=0.01)
    return tf.get_variable('W_' + name,
                           dtype=tf.float32,
                           shape=shape,
                           initializer=initer)

def bias_variable(name, shape):
    """
    Create a bias variable with appropriate initialization
    :param name: bias variable name
    :param shape: bias variable shape
    :return: initialized bias variable
    """
    initial = tf.constant(0., shape=shape, dtype=tf.float32)
    return tf.get_variable('b_' + name,
                           dtype=tf.float32,
                           initializer=initial)

def fc_layer(x, num_units, name, use_relu=True, randomize = True, sigma=5):
    """
    Create a fully-connected layer
    :param x: input from previous layer
    :param num_units: number of hidden units in the fully-connected layer
    :param name: layer name
    :param use_relu: boolean to add ReLU non-linearity (or not)
    :return: The output array
    """
    in_dim = x.get_shape()[1]
    W = weight_variable(name, shape=[in_dim, num_units])
    b = bias_variable(name, [num_units])
    layer = tf.matmul(x, W, name='Act_' + name)
    layer += b
    if use_relu:
        layer = tf.nn.relu(layer,name='Act_Relu_' + name)
    if randomize:
    	with tf.name_scope('Randomize_' + name):
    		# multiplies the sigma by the mean so it doesn't just take over... which would be realy annoying, though realistically batch norm should perhaps take care of this!
    		rand = tf.random_normal([num_units], tf.constant(0, tf.float32), tf.reduce_mean(layer) * tf.constant(sigma, tf.float32), name="randomize")
    	with tf.name_scope("rand_activations_" + name):
    		layer = tf.add(rand, layer, name='Randomized_layer_' + name) # hopefully this wll work!
    		#tf.Print(layer)
    return layer

    # this seems to work which is fantastic... I should adjust the size of sigma to the main thing though!

# try randomizing weights... first I need to actually print out the outputs... somehow!
# to check that it is actually randomizing... I'm not totally sure how to do this... dagnabbit!
# this is good... it doesreally seem to hurt except by massively messing up/degrading training, but that's good
# could easily just adjust it to the average shape of the activations... ugh!

# well... it sort of works... but who knows
 

# Create the graph for the linear model
# Placeholders for inputs (x) and outputs(y)
#learning_rate = tf.placeholder(tf.float32, shape=[])
learning_rate = tf.Variable(tf.float32, 0.01)
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='X')
y = tf.placeholder(tf.float32, shape=[None, n_classes], name='Y')
 

fc1 = fc_layer(x, h1, 'FC1', use_relu=True)
output_logits = fc_layer(fc1, n_classes, 'OUT', use_relu=False, randomize = False)

# Network predictions
cls_prediction = tf.argmax(output_logits, axis=1, name='predictions')

# Define the loss function, optimizer, and accuracy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_logits), name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam-op')
train_step = optimizer.minimize(loss)
correct_prediction = tf.equal(tf.argmax(output_logits, 1), tf.argmax(y, 1), name='correct_pred')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
 
 # Create the op for initializing all variables
init = tf.global_variables_initializer()
 
sess = tf.InteractiveSession()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
sess.run(init)
global_step = 0
# Number of training iterations in each epoch
num_tr_iter = int(len(y_train) / batch_size)
for epoch in range(epochs):
    print('Training epoch: {}'.format(epoch + 1))
    x_train, y_train = randomize(x_train, y_train)
    for iteration in range(num_tr_iter):
        global_step += 1
        start = iteration * batch_size
        end = (iteration + 1) * batch_size
        x_batch, y_batch = get_next_batch(x_train, y_train, start, end)

        # Run optimization op (backprop)
        learning_rate = 1e10
        # well that works and is nice... not sure if they even use feed dicts.. but now I know how to flip activations
        # and just generally change everything... so tht's awesome!
        feed_dict_batch = {x: x_batch, y: y_batch, learning_rate: lr}
        #print(dir(optimizer))
       	#print(dir(optimizer))
        #print(optimizer._lr_t.eval())
        #optimizer._lr = 1e20
        #optimizer._lr_t = tf.constant(1e20, tf.float32)
       # optimizer._lr = 1e20 # so this does work to change the learning rate internally... which is really nice
        # there has to be a better way thandoing this!
        #print(optimizer._lr)
        # let's just recreate optimizer 
        #optimizer = tf.train.AdamOptimizer(learning_rate = 1e20) # crap... this doesn't work either!
        # maybe I have to replace the whole train step?
        sess.run(train_step, feed_dict=feed_dict_batch)
        #print(optimizer._lr)
        # this does not sem to be doing anything though... dagnabbit!
        # okay.. none of this appears to be having any major effect... let's just recreate the optimizer each time
        # that's awesome... I can set learning rate to be a tensor and thus stochastically mess it up!
        # and have learning rate as a placeholder... let's try that!
        if iteration % display_freq == 0:
            # Calculate and display the batch loss and accuracy
            loss_batch, acc_batch = sess.run([loss, accuracy],
                                             feed_dict=feed_dict_batch)

            print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".
                  format(iteration, loss_batch, acc_batch))

    # Run validation after every epoch
    feed_dict_valid = {x: x_valid[:1000], y: y_valid[:1000]}
    loss_valid, acc_valid = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
    print('---------------------------------------------------------')
    print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".
          format(epoch + 1, loss_valid, acc_valid))
    print('---------------------------------------------------------')


    # let's also  try stochastic learning rates!
 

# set up really simple fc layer!

# so aims are to test stochastic activations, and stochastic learning rates... which both require slightly different remedies!

# I'm fairly unsure of how their method works... but it applies at the graph building stage... which might be correct
# and theoretically I can probably figure that out... the key is that the trainable variables are essentially the weights and the biases
# while the non-trainable variables are actually the activations... so that doesn't necessarily help... it might do however
# so I could just test it with defining my own versions of ops if I really want to to see if that helps at all
# but most likely it will simply be noise for the optimizer and ignored... but it might work... the only way to tell is empirical!
# so I can implement new ops for that without significant issue

# the learning rate is a different one, and I can have change learnign rate either within epoch or within each batch
# and will need to test those differences, but if it works it will be fantastic!
# so I just need to alter the current code to account for those things and see what happens there!

# let's test stochastic activations!
# so this does seem to work... which is really nice... the learning rates are simple to achieve as well.... if I want to!
# I just have to recreate the optimizer every single tmie... which is not ideal... but who knows!