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
def test_net(x, name, num_layers=2, hidden_units=32, isCNN=True, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        model = MyModel(x,name)
        if (isCNN):
            model.add(tf.reshape(x, shape=[-1, 28, 28, 1], name="input"))
            for l in range(num_layers):
                model.add(conv2d_layer(model.outputs, 
                                     filter_shape=[3,3,model.outputs.get_shape().as_list()[3], hidden_units],
                                     strides=[1,1,1,1],
                                     name="conv" + str(l+1)
                                    )
                         )

            dim = np.prod(model.outputs.get_shape().as_list()[1:])
            model.add(tf.reshape(model.outputs, [-1, dim], name='flatten'))
        else:
            for l in range(num_layers):
                dim = np.prod(model.outputs.get_shape().as_list()[1:])
                model.add(dense_layer(model.outputs, dim,hidden_units, name="fc"+str(l+1)))

        dim = np.prod(model.outputs.get_shape().as_list()[1:])
        model.add(dense_layer(model.outputs, dim, num_classes, name="output", activation=tf.identity))
        model.summary()
        return model.outputs



if __name__=='__main__':

    mnist = input_data.read_data_sets("./MNIST/MNIST-data", one_hot=True)

# Training Parameters
    learning_rate = 0.001
    num_epoch = 50
    batch_size = 1024
    isCNN = True    # True for CNN model; False for DNN model

# Network Parameters
    num_input = 784 # MNIST data input (img shape: 28*28)
    num_classes = 10 # MNIST total classes (0-9 digits)
    #dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    #keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

    record_loss=[]
    record_grad = []
    step_sum = 0

    if (isCNN):
        model_name = "CNN_hidden"
        exp_type = [(2,2)]
    else:
        model_name = "DNN_hidden"
        exp_type = [(4,8)]


    for exp in exp_type:
# Construct model
        print("")
        print('\033[5;31;40mExperiment setting:\033[0m')
        print(model_name+str(exp[0]) + "_dim" + str(exp[1]))

        logits = test_net(X,name=model_name,
                          num_layers=exp[0],
                          hidden_units=exp[1],
                          isCNN=isCNN
                         )
        prediction = tf.nn.softmax(logits)

        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)


        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        grads_and_vars = optimizer.compute_gradients(loss_op,tf.trainable_variables())

        init = tf.global_variables_initializer()

        bar_length = 20

# Start training
        with tf.Session() as sess:

            sess.run(init)
            epoch_loss = []
            epoch_grad = []
            for epoch in range(num_epoch):
                print("")
                print("Epoch:{}/{}".format(epoch+1, num_epoch))

                total_steps = int(mnist.train.num_examples/batch_size) 
                for step in range(total_steps):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    # Run optimization op (backprop)
                    t = time.time()
                    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                    remain_time = round((time.time()-t)*(total_steps-step))
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})

                    sum = 0.0
                    
                    gradients_and_variables = sess.run(grads_and_vars, feed_dict={X: batch_x, Y: batch_y})
                    for g, v in gradients_and_variables:
                        if g is not None:
                            '''print("***this is variable***")
                            print(v.shape)
                            print(v)
                            print("***this is gradient***")
                            print(g.shape)
                            print(g)'''
                            sum = sum + (np.array(g)**2).sum()
                    grads = np.sqrt(sum)
                    

                    epoch_loss.append(loss)
                    epoch_grad.append(grads)

                    progress = round(bar_length*(step/total_steps))
                    text = "\rProgress: [%s] - ETA: %-4ds - loss: %-5.3f - acc: %-5.3f"%(
                      '='*(progress-1)+'>'+'.'*(bar_length-progress),
                      remain_time,
                      loss,
                      acc
                      )
                    sys.stdout.write(text)
                    sys.stdout.flush()

                    step_sum = step_sum + 1


            print('')
            print("Optimization Finished!")

            # Calculate accuracy for 256 MNIST test images
            print("Testing Accuracy:", \
                    sess.run(accuracy, feed_dict={X: mnist.test.images[:256],
                                                  Y: mnist.test.labels[:256],
                                             }))

            record_loss.append(epoch_loss)
            record_grad.append(epoch_grad)


    index = list(range(1, num_epoch+1))
    step = list(range(1,step_sum+1))
    for i in range(len(record_loss)):
        plt.figure()
        plt.subplot(211)
        if (isCNN):
            list_label = ["CNN_hidden2_dim8"]
        else:
            list_label = ["DNN_hidden2_dim8"]

        plt.plot(step, record_grad[i], label=list_label[i])

        pic_dir = './pic_1_2'
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)

        plt.xlabel("# of Iterations")
        plt.ylabel("Training Gradient Norm")
        #plt.title("Gradient Norm")
        plt.legend()

        plt.subplot(212)
        plt.plot(step, record_loss[i], label=list_label[i])

        plt.xlabel("# of Iterations")
        plt.ylabel("Training Loss")
        #plt.title("Loss")
        plt.legend()
        if (isCNN):
            plt.savefig(pic_dir+'/CNN'+str(i)+'.png')
        else:
            plt.savefig(pic_dir+'/DNN'+str(i)+'.png')