from keras.datasets import cifar10
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
import time
import sys
import os


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
        else:
            dim = np.prod(model.outputs.get_shape().as_list()[1:])
            model.add(tf.reshape(x, shape=[-1, dim], name="input"))
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

    num_classes = 10

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, num_classes).reshape(y_train.shape[0], -1)
    mydata = dataset(x_train, y_train)

# Training Parameters
    learning_rate = 0.001
    num_epoch = 50
    batch_size = 128
    isCNN = False    # True for CNN model; False for DNN model


    record_loss=[]
    record_acc= []

    if (isCNN):
        model_name = "CNN_hidden"
        exp_type = [(1,8), (1,16), (1,32), (2,8), (2,16), (4,8)]
    else:
        model_name = "DNN_hidden"
        exp_type = [(1,256), (1,512), (1,1024), (2,256), (2,512), (4,256)]


    for exp in exp_type:
# tf Graph input
        X = tf.placeholder(tf.float32, [None, 32,32,3])
        Y = tf.placeholder(tf.float32, [None, num_classes])
        #keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

# Construct model
        print("")
        print('\033[5;31;40mExperiment setting:\033[0m')
        print(model_name+str(exp[0]) + "_dim" + str(exp[1]))

        logits = test_net(X,name=model_name+str(exp[0]) + "_dim" + str(exp[1]),
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

        init = tf.global_variables_initializer()

        bar_length = 20

# Start training

        with tf.Session() as sess:

            sess.run(init)
            
            epoch_acc = []
            epoch_loss = []
            for epoch in range(num_epoch):
                print("")
                print("Epoch:{}/{}".format(epoch+1, num_epoch))
                temp_acc = []
                temp_loss = []
                total_steps = int(mydata.size/batch_size) 
                for step in range(total_steps):
                    batch_x, batch_y = mydata.next_batch(batch_size)
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

                epoch_acc.append(np.mean(temp_acc))
                epoch_loss.append(np.mean(temp_loss))

            print('')
            print("Optimization Finished!")

            record_acc.append(epoch_acc)
            record_loss.append(epoch_loss)

        tf.reset_default_graph()


    index = list(range(1, num_epoch+1))
    plt.figure()
    if (isCNN):
        list_label = ["CNN_hidden1_dim8", "CNN_hidden1_dim16", "CNN_hidden1_dim32",
                      "CNN_hidden2_dim8", "CNN_hidden2_dim16","CNN_hidden4_dim8"
                     ]
    else:
        list_label = ["DNN_hidden1_dim64", "DNN_hidden1_dim128", "DNN_hidden1_dim256",
                      "DNN_hidden2_dim64", "DNN_hidden2_dim128", "DNN_hidden4_dim64"
                     ]

    for i in range(len(record_acc)):
        plt.plot(index, record_acc[i], label=list_label[i])

    pic_dir = './pic'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    plt.xlabel("# of Epochs")
    plt.ylabel("Training Acc")
    plt.legend()
    if (isCNN):
        plt.savefig(pic_dir+'/CNN_acc_cifar.png')
    else:
        plt.savefig(pic_dir+'/DNN_acc_cifar.png')
    #plt.show()

    plt.figure()
    for i in range(len(record_acc)):
        plt.plot(index, record_loss[i], label=list_label[i])

    plt.xlabel("# of Epochs")
    plt.ylabel("Training Loss")
    plt.legend()
    if (isCNN):
        plt.savefig(pic_dir+'/CNN_loss_cifar.png')
    else:
        plt.savefig(pic_dir+'/DNN_loss_cifar.png')
    #plt.show()
