import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import sys
import os
import keras
from keras.utils import np_utils

class MyModel:

    def __init__(self, input_tensor, name, layers=None):
        self.layers = []            # Stack of layers.
        self.outputs = input_tensor
        self.name = name

        # Add to the model any layers passed to the constructor.
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        self.layers.append(layer)
        self.outputs = layer

    def summary(self):
        def print_row(fields, positions):
            line = ''
            for i in range(len(fields)):
                if i > 0:
                    line = line[:-1] + ' '
                line += str(fields[i])
                line = line[:positions[i]]
                line += ' ' * (positions[i] - len(line))
            print(line)

        line_length = 65;
        position = [.45, .85, 1.]
        if position[-1] <= 1:
            position = [int(line_length * p) for p in position]
        to_display = ['Layer (type)', 'Output Shape', 'Param #']

        print('_' * line_length)
        print_row(to_display, position)
        print('=' * line_length)
        total_cnt = 0

        for layer in tf.trainable_variables(scope=self.name):
            param_cnt = np.prod(layer.get_shape().as_list())
            total_cnt += param_cnt
            output_shape = layer.get_shape().as_list()
            fields = [layer.name, output_shape, param_cnt]
            print_row(fields, position)

        print('='*line_length)
        print("Total parameters:", total_cnt)
        print('')

        return total_cnt


# Create some wrappers for simplicity
def conv2d_layer(input_tensor, filter_shape, strides, name, padding='SAME', activation=tf.nn.relu):
    with tf.variable_scope(name):
        filters = tf.get_variable('filters', filter_shape, dtype=tf.float32)
        bias = tf.get_variable('bias', [filter_shape[3]], dtype=tf.float32)
        conv = tf.nn.conv2d(input_tensor, filters, strides, padding)
        output = activation(tf.nn.bias_add(conv, bias))
        return output


def maxpool2d(input_tensor, name, k=2):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def dense_layer(input_tensor, input_dim, output_dim, name, activation=tf.nn.relu):
    with tf.variable_scope(name):
        W = tf.get_variable('weights', [input_dim, output_dim], dtype=tf.float32)
        b = tf.get_variable('bias', [output_dim], dtype=tf.float32)
        output = tf.nn.bias_add(tf.matmul(input_tensor, W), b)
        return activation(output)


# Create model
def test_net(x, name, num_layers=2, hidden_units=32, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        model = MyModel(x,name)

        model.add(tf.reshape(x, shape=[-1, 32, 32, 3], name="input"))
        for l in range(num_layers):
            model.add(conv2d_layer(model.outputs, 
                                 filter_shape=[3,3,model.outputs.get_shape().as_list()[3], hidden_units],
                                 strides=[1,1,1,1],
                                 name="conv" + str(l+1)
                                )
                     )
            #model.add(maxpool2d(model.outputs, name="maxpool"+str(l+1)))

        dim = np.prod(model.outputs.get_shape().as_list()[1:])
        model.add(tf.reshape(model.outputs, [-1, dim], name='flatten'))

        dim = np.prod(model.outputs.get_shape().as_list()[1:])
        model.add(dense_layer(model.outputs, dim, num_classes, name="output", activation=tf.identity))
        count = model.summary()
        return model.outputs, count



if __name__=='__main__':
    # Network Parameters
    num_train_examples=50000
    num_test_examples=10000
    img_channels = 3
    num_input = 1024 # cifar10 data input (img shape: 32*32)
    num_classes = 10 # cifar10 total classes
    #dropout = 0.75 # Dropout, probability to keep units

    # Training Parameters
    learning_rate = 0.001
    num_epoch = 30
    batch_size = 100
    test_batch_size = 1000

    # Load the dataset
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(Y_train, num_classes)
    print('Y_train shape:', Y_train.shape)
    Y_test = np_utils.to_categorical(Y_test, num_classes)

    batch_x=np.zeros((batch_size,32, 32,img_channels)).astype('float32')
    print('batch_x shape:', batch_x.shape)
    batch_y=np.zeros((batch_size,num_classes)).astype('float32')
    print('batch_y shape:', batch_y.shape)

    test_batch_x=np.zeros((test_batch_size,32, 32,img_channels)).astype('float32')
    test_batch_y=np.zeros((test_batch_size,num_classes)).astype('float32')

    record_parnum = []
    record_loss=[]
    record_acc= []
    record_test_acc = []
    record_test_loss = []

    model_name = "CNN_hidden"
    exp_type = [(2,2),(2,4),(2,6),(4,8),(4,10),(4,12),(4,14),(8,16),(8,20),(10,22),(10,24)]

    for exp in exp_type:
        acc = 0.0
        loss = 0.0
# tf Graph input
        X = tf.placeholder(tf.float32, [None, 32,32,3])
        Y = tf.placeholder(tf.float32, [None, num_classes])
        #keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Construct model
        print("")
        print('\033[5;31;40mExperiment setting:\033[0m')
        print(model_name+str(exp[0]) + "_dim" + str(exp[1]))

        logits, cnt = test_net(X,name=model_name+str(exp[0]) + "_dim" + str(exp[1]),
                          num_layers=exp[0],
                          hidden_units=exp[1]
                         )
        record_parnum.append(cnt)

        prediction = tf.nn.softmax(logits)

        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)


        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()

        bar_length = 20

# Start training

        with tf.Session() as sess:

            sess.run(init)

            for epoch in range(num_epoch):
                print("")
                print("Epoch:{}/{}".format(epoch+1, num_epoch))
                temp_acc = []
                temp_loss = []
                total_steps = int(num_train_examples/batch_size) 
                for step in range(total_steps):
                    for j in range(batch_size):
                        batch_x[j] = X_train[step*batch_size+j]
                        batch_y[j] = Y_train[step*batch_size+j]
                    # Run optimization op (backprop)
                    t = time.time()
                    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                    remain_time = round((time.time()-t)*(total_steps-step))
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})

                    temp_acc.append(acc)
                    temp_loss.append(loss)

                    progress = round(bar_length*(step/total_steps))
                    text = "\rProgress: [%s] - ETA: %-4ds - loss: %-5.3f - acc: %-5.3f"%(
                      '='*(progress-1)+'>'+'.'*(bar_length-progress),
                      remain_time,
                      loss,
                      acc
                      )
                    sys.stdout.write(text)
                    sys.stdout.flush()

                acc = np.mean(temp_acc)
                loss = np.mean(temp_loss)

            print('')
            print("train acc:{}".format(acc))
            print("train loss:{}".format(loss))
            print("Optimization Finished!")


            tmp_loss = [] #for test
            tmp_acc = [] #for test
            test_steps = int(num_test_examples/test_batch_size) 
            for i in range(test_steps):
                for j in range(test_batch_size):
                    test_batch_x[j] = X_train[i*test_batch_size+j]
                    test_batch_y[j] = Y_train[i*test_batch_size+j]
                test_loss, test_acc = sess.run( [loss_op, accuracy],
                    feed_dict={X: test_batch_x, Y: test_batch_y})
                tmp_loss.append(test_loss)
                tmp_acc.append(test_acc)
            print("Testing Accuracy:{}".format(np.mean(tmp_acc)))
            print("test loss:{}".format(np.mean(tmp_loss)))

            record_acc.append(acc)
            record_loss.append(loss)
            record_test_acc.append(np.mean(test_acc))
            record_test_loss.append(np.mean(test_loss))

        tf.reset_default_graph()


    plt.figure()
    plt.scatter(record_parnum, record_acc, label = "train_acc")
    plt.scatter(record_parnum, record_test_acc, label = "test_acc")

    pic_dir = './pic'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    plt.xlabel("# of Parameters")
    plt.ylabel("Acc")
    plt.legend()

    plt.savefig(pic_dir+'/CNN_acc.png')


    plt.figure()
    plt.scatter(record_parnum, record_loss, label = "train_loss")
    plt.scatter(record_parnum, record_test_loss, label = "test_loss")

    plt.xlabel("# of Parameters")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(pic_dir+'/CNN_loss.png')

