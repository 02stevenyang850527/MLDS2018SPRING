import os
import math
import time
import argparse
import numpy as np
import tensorflow as tf

from model import medium
from utils import shuffle, batch, stdout, tfmt


parser = argparse.ArgumentParser(prog='hw1_2_3.py', description='MLDS2018 hw1-2.3 Minimal Ratio')
parser.add_argument('--sr', type=int, default=1000)
parser.add_argument('--rng', type=list, default=[0., 3.])

parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epoch', type=int, default=5000)
parser.add_argument('--batch', type=int, default=-1)

parser.add_argument('--prefix', type=str, default='minimal_ratio')
parser.add_argument('--save_dir', type=str, default='./')
parser.add_argument('--result_dir', type=str, default='./')
parser.add_argument('--result_file', type=str, default=None)
args = parser.parse_args()


# construct stimulated function
def target_func(x):
    return math.exp(-x) * math.cos(2 * math.pi * x)

tr_x = np.arange(*args.rng, 1 / args.sr, dtype=np.float32)
tr_y = np.array([target_func(x) for x in tr_x], dtype=np.float32)
tr_x = tr_x.reshape(-1, 1)
tr_y = tr_y.reshape(-1, 1)


# construct model
X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])
out = medium(X, 'medium', 4, verbose=1)

mse = tf.losses.mean_squared_error(Y, out)
var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='medium')

grads = tf.gradients(mse, var)
grad_norm = tf.norm([tf.norm(grad) for grad in grads])

train_step1 = tf.train.AdamOptimizer(args.lr).minimize(mse)
train_step2 = tf.train.AdamOptimizer(args.lr).minimize(grad_norm)


# start training
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_steps = max(np.ceil(len(tr_x) / args.batch), 1)
    
    print('Minimize Mean Squared Error.')

    for n_epoch in range(args.epoch):
        cost = 0
        n_step = 0
        rdn_x, rdn_y = shuffle(tr_x, tr_y)

        for batch_x, batch_y in batch(rdn_x, rdn_y, size=args.batch):
            tStart = time.time()
            sess.run(train_step1, feed_dict={X: batch_x, Y: batch_y})
            step_loss = sess.run(mse, feed_dict={X: batch_x, Y: batch_y})
            step_norm = sess.run(grad_norm, feed_dict={X: batch_x, Y: batch_y})
            tEnd = time.time()
             
            if n_epoch % 50 == 49:
                stdout(n_epoch, n_step, total_steps, tEnd-tStart, step_loss, grad_norm=step_norm)
            cost += tEnd - tStart
            n_step += 1

        epoch_loss = sess.run(mse, feed_dict={X: tr_x, Y: tr_y})
        epoch_norm = sess.run(grad_norm, feed_dict={X: tr_x, Y: tr_y})
        if n_epoch % 50 == 49:
            stdout(n_epoch, n_step, total_steps, cost, epoch_loss, grad_norm=epoch_norm)
    

    print('\nMinimize Gradient Norm.')
    n_epoch = 0
    epoch_norm = np.finfo(dtype=np.float32).max

    while(epoch_norm > 1e-5):
        cost = 0
        n_step = 0
        rdn_x, rdn_y = shuffle(tr_x, tr_y)

        for batch_x, batch_y in batch(rdn_x, rdn_y, size=args.batch):
            tStart = time.time()
            sess.run(train_step2, feed_dict={X: batch_x, Y: batch_y})
            step_loss = sess.run(mse, feed_dict={X: batch_x, Y: batch_y})
            step_norm = sess.run(grad_norm, feed_dict={X: batch_x, Y: batch_y})
            tEnd = time.time()
             
            if n_epoch % 500 == 499:
                stdout(n_epoch, n_step, total_steps, tEnd-tStart, step_loss, grad_norm=step_norm)
            cost += tEnd - tStart
            n_step += 1

        epoch_loss = sess.run(mse, feed_dict={X: tr_x, Y: tr_y})
        epoch_norm = sess.run(grad_norm, feed_dict={X: tr_x, Y: tr_y})
        if n_epoch % 500 == 499:
            stdout(n_epoch, n_step, total_steps, cost, epoch_loss, grad_norm=epoch_norm)
        
        n_epoch += 1
        if n_epoch == 50000: os._exit(0)
   
    #pred = sess.run(out, feed_dict={X: tr_x})
    
    
    print('\nCalculate Hessian Matrix and Minimal Ratio.', end=' ')

    tStart = time.time()
    minimal_ratio = 0
    hess_mat = []

    for grad in grads:
        flat_grad = tf.reshape(grad, (-1,))
        for i in range(flat_grad.shape[0]):
            elemwise_grad = tf.slice(flat_grad, [i], [1])
            hess_row = tf.gradients(elemwise_grad, var)
            hess_row = tf.concat([tf.reshape(elem, (-1,)) for elem in hess_row], axis=0)
            hess_mat.append(hess_row)
    
    for x, y in zip(tr_x, tr_y):
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        hess = sess.run(hess_mat, feed_dict={X: x , Y: y})
        hess = np.stack(hess, axis=0)
        eigvals = np.linalg.eigvals(hess)
        minimal_ratio += sum([(eigval > 0) for eigval in eigvals]) / len(eigvals)

    minimal_ratio /= len(tr_x)
    tEnd = time.time()
    
    print('cost {:s}'.format(tfmt(tEnd-tStart)))


# save Prediction and loss
if os.path.exists(args.save_dir + args.prefix + '_loss.csv'):
    lossfile = open(args.save_dir + args.prefix + '_loss.csv', 'a')
else:
    lossfile = open(args.save_dir + args.prefix + '_loss.csv', 'w')
    print('loss,minimal ratio,gradient norm', file=lossfile)
print(epoch_loss, minimal_ratio, epoch_norm, sep=',', file=lossfile)
lossfile.close()

'''
if args.result_file: outfile = open(args.result_file, 'w')
else: outfile = open(args.result_dir + args.prefix + '.csv', 'w')
tr_x, pred = tr_x.squeeze(), pred.squeeze()
print('x,f(x)', file=outfile)
for x, p in zip(tr_x, pred): print(x, p, sep=',', file=outfile)
outfile.close()
'''
