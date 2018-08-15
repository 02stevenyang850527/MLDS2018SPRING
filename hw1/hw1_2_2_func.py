import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import sys
import os

# Network Parameters
num_input = 1 
num_classes = 1
n_hidden = [[5,10,5,1]]
#[200, 1], [10,18,15,4,1],

# Define the neural network
def neural_net(x, par):
    tmp = tf.layers.dense(x, par[0])
    for i in range(len(par)-1):
        tmp = tf.layers.dense(tmp, par[i+1])
    return tmp

if __name__=='__main__':

# Setting input data
    #x = np.array([[random.random()] for _ in range(100)])
    x = 0.01*(np.arange(100).reshape(100,1))
    np.random.shuffle(x)
    print(x)
    y = np.sinc(5*(np.pi)*x)

    test_x = np.array([[random.random()] for _ in range(20)])
    test_y = np.sinc(5*(np.pi)*test_x)

# Training Parameters
    learning_rate = 0.0003
    num_epoch = 150
    batch_size = 10

# tf Graph input
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    #keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

    record_loss=[]
    test_result = []

    model_name = ["DNN", "DNN_medium", "DNN_deep"]

    model_count = 1
    steps = 0


    for cnt in range(model_count):
# Construct model
        print("")
        print('\033[5;31;40mExperiment setting:\033[0m')
        print(model_name[cnt])
        
        logits = neural_net(X, n_hidden[cnt])

        loss_op = tf.losses.mean_squared_error(Y, logits)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)
        
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
                total_steps = int(100/batch_size) 
                for step in range(total_steps):
                    steps += 1
                    sum = 0
                    batch_x = np.array(x[step:step+10])
                    batch_y = np.array(y[step:step+10])

                    # Run optimization op (backprop)
                    _, loss = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
                    epoch_loss.append(loss)

                    grads_and_vars = optimizer.compute_gradients(loss_op,tf.trainable_variables())
                    gradients_and_variables = sess.run(grads_and_vars, feed_dict={X: batch_x, Y: batch_y})
                    for g, v in gradients_and_variables:
                        if g is not None:
                            '''print("***this is variable***")
                            print(v.shape)
                            print(v)
                            print("***this is gradient***")
                            print(g.shape)
                            print(g)
                            print((np.array(g)**2).sum())'''
                            sum = sum + (np.array(g)**2).sum()
                    grads = np.sqrt(sum)
                    epoch_grad.append(grads)

                    progress = round(bar_length*(step/total_steps))
                    text = "\rProgress: [%s] - loss: %-5.3f "%(
                      '='*(progress-1)+'>'+'.'*(bar_length-progress),
                      loss
                      )
                    sys.stdout.write(text)
                    sys.stdout.flush()

            print('')
            print("Optimization Finished!")


    index = list(range(1, num_epoch+1))
    iteration = list(range(1,steps+1))

    list_label = ["DNN_hidden1_small", "DNN_hidden1_medium", "DNN_hidden1_deep"]

    pic_dir = './pic'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)


    plt.figure()
    plt.subplot(211)
    plt.plot(iteration, epoch_loss, label="DNN_function")

    #plt.xlabel("# of Iterations")
    plt.ylabel("Training Loss")
    #plt.title("Loss")
    plt.legend()

    plt.subplot(212)
    plt.plot(iteration, epoch_grad, label="DNN_function")

    plt.xlabel("# of Iterations")
    plt.ylabel("Training Gradient")
    #plt.title("Gradient")
    plt.legend()

    plt.savefig(pic_dir+'/DNN_loss.png')
#plt.show()