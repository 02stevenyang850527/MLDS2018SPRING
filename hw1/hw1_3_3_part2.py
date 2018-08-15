import os
import time
import argparse
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from model import cnn
from utils import shuffle, batch, stdout


parser = argparse.ArgumentParser(prog='hw1_3_3_part2.py', description='MLDS2018 hw1-1.3 part2 Sensitivity')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--batch', type=int, default=-1)
parser.add_argument('--early_stop', type=int, default=10)

parser.add_argument('--prefix', type=str, default='sensitivity')
parser.add_argument('--save_dir', type=str, default='./')
parser.add_argument('--result_dir', type=str, default='./')
parser.add_argument('--result_file', type=str, default=None)
args = parser.parse_args()


# import mnist data
mnist = input_data.read_data_sets('./MNIST/', one_hot=True)


# construct model
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
out = cnn(X, 'cnn', 32)

loss = tf.losses.softmax_cross_entropy(Y, out)
if   args.opt == 'sgd' : train_step = tf.train.GradientDescentOptimizer(args.lr).minimize(loss)
elif args.opt == 'adam': train_step = tf.train.AdamOptimizer(args.lr).minimize(loss)

jacobian = []
for i in range(out.shape[1]):
    jacobian.append(tf.gradients(out[:, i], X)[0])

# start training
init = tf.global_variables_initializer()

var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='cnn')
saver = tf.train.Saver(var)

with tf.Session() as sess:
    sess.run(init)
    n_epoch = 0
    stop_cnt = 0
    best_loss = np.finfo(dtype=np.float32).max
    total_steps = max(int(np.ceil(mnist.train.num_examples / args.batch)), 1)
    
    while(stop_cnt < args.early_stop):
        step_loss = []
        step_acc = []
        cost = 0
             
        for n_step in range(total_steps):
            tStart = time.time()
            batch_x, batch_y = mnist.train.next_batch(args.batch)
            sess.run(train_step, feed_dict={X: batch_x, Y: batch_y})
            step_loss.append(sess.run(loss, feed_dict={X: batch_x, Y: batch_y}))
            
            pred = sess.run(out, feed_dict={X: batch_x, Y: batch_y})
            correct = sum(np.equal(np.argmax(pred, 1), np.argmax(batch_y, 1)))
            step_acc.append(correct / pred.shape[0])
            tEnd = time.time()
             
            stdout(n_epoch, n_step, total_steps, tEnd-tStart, step_loss[-1], acc=step_acc[-1])
            cost += tEnd - tStart
        
        epoch_loss = np.mean(step_loss)
        epoch_acc = np.mean(step_acc)
        stdout(n_epoch, total_steps, total_steps, cost, epoch_loss, acc=epoch_acc)

        if best_loss > epoch_loss:
            saver.save(sess, args.save_dir + args.prefix + '_{}.ckpt'.format(args.batch)) 
            best_loss = epoch_loss
            stop_cnt = 0
        else: stop_cnt += 1

        n_epoch += 1
     

    saver.restore(sess, args.save_dir + args.prefix + '_{}.ckpt'.format(args.batch)) 

    train_cnt = 0
    train_loss = 0

    for i in range(0, mnist.train.num_examples, 256):
        batch_x = mnist.train.images[i:i+256]
        batch_y = mnist.train.labels[i:i+256]
        train_loss += sess.run(loss, feed_dict={X: batch_x, Y: batch_y}) * batch_x.shape[0]

        pred = sess.run(out, feed_dict={X: batch_x, Y: batch_y})
        train_cnt += sum(np.equal(np.argmax(pred, 1), np.argmax(batch_y, 1)))

    train_loss = train_loss / mnist.train.num_examples
    train_acc = train_cnt / mnist.train.num_examples


    test_cnt = 0
    test_loss = 0
    sensitivity = 0

    for i in range(0, mnist.test.num_examples, 256):
        batch_x = mnist.test.images[i:i+256]
        batch_y = mnist.test.labels[i:i+256]
        test_loss += sess.run(loss, feed_dict={X: batch_x, Y: batch_y}) * batch_x.shape[0]

        pred = sess.run(out, feed_dict={X: batch_x, Y: batch_y})
        test_cnt += sum(np.equal(np.argmax(pred, 1), np.argmax(batch_y, 1)))
        
        jaco_mats = sess.run(jacobian, feed_dict={X: batch_x, Y: batch_y})
        jaco_mats = np.stack(jaco_mats, 1)
        sensitivity += np.sum([np.linalg.norm(jaco_mat) for jaco_mat in jaco_mats])
    
    sensitivity = sensitivity / mnist.test.num_examples
    test_loss = test_loss / mnist.test.num_examples
    test_acc = test_cnt / mnist.test.num_examples


# save prediction and loss
if os.path.exists(args.save_dir + args.prefix + '_loss.csv'):
    lossfile = open(args.save_dir + args.prefix + '_loss.csv', 'a')
else:
    lossfile = open(args.save_dir + args.prefix + '_loss.csv', 'w')
    print('batch,training loss,training accuracy,testing loss,testing accuracy,sensitivity', file=lossfile)
print(args.batch, train_loss, train_acc, test_loss, test_acc, sensitivity, sep=',', file=lossfile)
lossfile.close()

