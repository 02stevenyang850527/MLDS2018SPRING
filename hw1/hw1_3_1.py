import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import sys
import os

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

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


def dense_layer(input_tensor, input_dim, output_dim, name, activation=tf.nn.relu):
    with tf.variable_scope(name):
        W = tf.get_variable('weights', [input_dim, output_dim], dtype=tf.float32)
        b = tf.get_variable('bias', [output_dim], dtype=tf.float32)
        output = tf.nn.bias_add(tf.matmul(input_tensor, W), b)
        return activation(output)


def test_net(x, name, num_layers=3, hidden_units=256, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        model = MyModel(x,name)
        for l in range(num_layers):
            dim = np.prod(model.outputs.get_shape().as_list()[1:])
            model.add(dense_layer(model.outputs, dim,hidden_units, name="fc"+str(l+1)))

        dim = np.prod(model.outputs.get_shape().as_list()[1:])
        model.add(dense_layer(model.outputs, dim, num_classes, name="output", activation=tf.identity))
        model.summary()
        return model.outputs


class dataset:
    
    def __init__(self, x, y):
        self.cnt = 0;
        self.x = x;
        self.y = y;
        self.size = x.shape[0]

    def next_batch(self, batch_size):
        if self.cnt + batch_size > self.size:
            used = np.arange(self.cnt)
            np.random.shuffle(used)
            rest = np.arange(self.cnt, self.size)
            shuffle = np.append(rest, used)
            self.x = self.x[shuffle,:]
            self.y = self.y[shuffle,:]
            self.cnt = 0
        start = self.cnt
        self.cnt += batch_size
        end = self.cnt
        batch_x = self.x[start:end,:]
        batch_y = self.y[start:end,:]

        return batch_x, batch_y


if __name__=='__main__':
    
    mnist = input_data.read_data_sets("./MNIST", one_hot=True)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    index = np.random.permutation(mnist.train.num_examples)
    y_random = y_train[index]

    mydata = dataset(x_train, y_random)

    learning_rate = 0.001
    num_epoch = 300
    batch_size = 128

    num_input = 784 # MNIST data input (img shape: 28*28)
    num_classes = 10 # MNIST total classes (0-9 digits)


    X = tf.placeholder(tf.float32, [None, num_input], name="Input_image")
    Y = tf.placeholder(tf.float32, [None, num_classes], name="Output_label")

    model_name = "DNN"
    num_layers = 3
    num_hidden = 256
    train_loss = np.zeros(num_epoch)
    test_loss = np.zeros(num_epoch)
    
# Construct model
    print("")
    print('\033[5;31;40mExperiment setting:\033[0m')
    print(model_name)
    logits = test_net(X,name=model_name,
                      num_layers=num_layers,
                      hidden_units=num_hidden
                     )
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

    total_steps = int(mnist.train.num_examples/batch_size)

    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epoch):
            print("")
            print("Epoch:{}/{}".format(epoch+1, num_epoch))
            temp_loss = np.zeros(total_steps)

            for step in range(total_steps):
                batch_x, batch_y = mydata.next_batch(batch_size)
                index = np.random.permutation(batch_size)
                batch_y_random = batch_y[index]

                # Run optimization op (backprop)
                t = time.time()
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                remain_time = round((time.time()-t)*(total_steps-step))
                temp_loss[step], acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})

                progress = round(bar_length*(step/total_steps))
                text = "\rProgress: [%s] - ETA: %-4ds - loss: %-5.3f - acc: %-5.3f"%(
                  '='*(progress-1)+'>'+'.'*(bar_length-progress),
                  remain_time,
                  temp_loss[step],
                  acc
                  )
                sys.stdout.write(text)
                sys.stdout.flush()

            train_loss[epoch] = temp_loss.mean()
            test_loss[epoch] = sess.run(loss_op, feed_dict={X:mnist.test.images,
                                                         Y:mnist.test.labels
                                                        })
    print("")
    pic_dir = './pic'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    plt.plot(list(range(1, num_epoch+1)), train_loss, label='train_loss')
    plt.plot(list(range(1, num_epoch+1)), test_loss, label='test_loss')

    plt.xlabel("# of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(pic_dir+'/DNN_shuffle.png')
    plt.show()
