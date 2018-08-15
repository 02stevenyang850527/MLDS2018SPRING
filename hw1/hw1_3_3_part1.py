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


def test_net(x, name, num_layers=2, hidden_units=32, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        model = MyModel(x,name)
        for l in range(num_layers):
            dim = np.prod(model.outputs.get_shape().as_list()[1:])
            model.add(dense_layer(model.outputs, dim,hidden_units, name="fc"+str(l+1)))

        dim = np.prod(model.outputs.get_shape().as_list()[1:])
        model.add(dense_layer(model.outputs, dim, num_classes, name="output", activation=tf.identity))
        model.summary()
        return model.outputs


if __name__=='__main__':
    mnist = input_data.read_data_sets("./MNIST", one_hot=True)
    num_input = 784 
    num_classes = 10
    learning_rate = 0.001
    num_epoch = 3
    exp_type = [64, 1024]
    model_name = "DNN_hidden2_dim128_batch"
    weight = {}
    bias = {}

    for batch_size in exp_type:
        X = tf.placeholder(tf.float32, [None, num_input])
        Y = tf.placeholder(tf.float32, [None, num_classes])

        print("")
        print('\033[5;31;40mExperiment setting:\033[0m')
        print(model_name+str(batch_size))

        logits = test_net(X,name=model_name+str(batch_size),
                          num_layers=2,
                          hidden_units=128,
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

        with tf.Session() as sess:

            sess.run(init)
            
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

                    progress = round(bar_length*(step/total_steps))
                    text = "\rProgress: [%s] - ETA: %-4ds - loss: %-5.3f - acc: %-5.3f"%(
                      '='*(progress-1)+'>'+'.'*(bar_length-progress),
                      remain_time,
                      loss,
                      acc
                      )
                    sys.stdout.write(text)
                    sys.stdout.flush()

            weight["wf1_"+str(batch_size)] = sess.run(tf.trainable_variables(scope=model_name+str(batch_size)+'/fc1/weights:0')),
            weight["wf2_"+str(batch_size)] = sess.run(tf.trainable_variables(scope=model_name+str(batch_size)+'/fc2/weights:0')),
            weight["wfo_"+str(batch_size)] = sess.run(tf.trainable_variables(scope=model_name+str(batch_size)+'/output/weights:0')),

            bias["bf1_"+str(batch_size)] = sess.run(tf.trainable_variables(scope=model_name+str(batch_size)+'/fc1/bias:0')),
            bias["bf2_"+str(batch_size)] = sess.run(tf.trainable_variables(scope=model_name+str(batch_size)+'/fc2/bias:0')),
            bias["bfo_"+str(batch_size)] = sess.run(tf.trainable_variables(scope=model_name+str(batch_size)+'/output/bias:0')),


            print('')
            print("Optimization Finished!")


    print("Training Completed")


    print("Start Interpolation Experiment")

    sample_point = 300
    alpha_set = np.linspace(-1,2,sample_point)
    test_loss_record = np.zeros(sample_point)
    test_acc_record = np.zeros(sample_point)
    train_loss_record = np.zeros(sample_point)
    train_acc_record = np.zeros(sample_point)
    weight_test = {}
    bias_test = {}
    total_steps = int(mnist.train.num_examples/batch_size)
    
    for index, alpha in enumerate(alpha_set):
        progress = round(bar_length*(index/sample_point))
        text = "\rProgress: [%s]  %d/%d"%(
          '='*(progress-1)+'>'+'.'*(bar_length-progress),
          index+1,
          sample_point
        )
        sys.stdout.write(text)
        sys.stdout.flush()


        tf.reset_default_graph()
        weight_test['wf1'] = (1.0-alpha)*weight['wf1_'+str(exp_type[0])][0][0] + alpha*weight['wf1_'+str(exp_type[1])][0][0]
        weight_test['wf2'] = (1.0-alpha)*weight['wf2_'+str(exp_type[0])][0][0] + alpha*weight['wf2_'+str(exp_type[1])][0][0]
        weight_test['wfo'] = (1.0-alpha)*weight['wfo_'+str(exp_type[0])][0][0] + alpha*weight['wfo_'+str(exp_type[1])][0][0]

        bias_test['bf1'] = (1.0-alpha)*bias['bf1_'+str(exp_type[0])][0][0] + alpha*bias['bf1_'+str(exp_type[1])][0][0]
        bias_test['bf2'] = (1.0-alpha)*bias['bf2_'+str(exp_type[0])][0][0] + alpha*bias['bf2_'+str(exp_type[1])][0][0]
        bias_test['bfo'] = (1.0-alpha)*bias['bfo_'+str(exp_type[0])][0][0] + alpha*bias['bfo_'+str(exp_type[1])][0][0]

        X = tf.placeholder(tf.float32, [None, num_input])
        Y = tf.placeholder(tf.float32, [None, num_classes])

        fc1 = tf.add(tf.matmul(X, weight_test['wf1']), bias_test['bf1'])
        fc1 = tf.nn.relu(fc1)
        fc2 = tf.add(tf.matmul(fc1, weight_test['wf2']), bias_test['bf1'])
        fc2 = tf.nn.relu(fc2)
        logits = tf.add(tf.matmul(fc2, weight_test['wfo']), bias_test['bfo'])

        prediction = tf.nn.softmax(logits)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                logits=logits, labels=Y))
        init = tf.global_variables_initializer()

        tmp_acc = []
        tmp_loss = []
        with tf.Session() as sess:
            sess.run(init)
            for step in range(total_steps):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                tmp_acc.append(acc)
                tmp_loss.append(loss)

            test_loss, test_acc = sess.run([loss_op, accuracy], feed_dict={X: mnist.test.images,
                                                                           Y: mnist.test.labels,
                                                                          })
            test_loss_record[index] = test_loss
            test_acc_record[index] = test_acc
            train_loss_record[index] = np.mean(tmp_loss)
            train_acc_record[index] = np.mean(tmp_acc)

    print('')
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(alpha_set, np.log(train_loss_record), '--b')
    p1, = ax1.plot(alpha_set, np.log(test_loss_record), 'b')
    ax1.set_ylabel('Cross entropy (log scale)')
    ax1.set_xlabel('alpha')
    ax1.yaxis.label.set_color('blue')
    ax1.tick_params(axis='y', colors=p1.get_color())

    ax2 = ax1.twinx()  # this is the important function
    ax2.plot(alpha_set, train_acc_record, '--r', label="Train")
    p2, = ax2.plot(alpha_set, test_acc_record, 'r', label="Test")
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('alpha')
    ax2.yaxis.label.set_color('red')
    ax2.tick_params(axis='y', colors=p2.get_color())
    plt.legend()

    pic_dir = './pic'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)
    plt.savefig(pic_dir+'/FlatenessVsGeneralization.png')
    plt.show()
