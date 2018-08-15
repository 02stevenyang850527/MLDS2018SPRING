import math
import time
import argparse
import numpy as np
import tensorflow as tf

from model import shallow, medium, deep
from utils import shuffle, batch, stdout


parser = argparse.ArgumentParser(prog='hw1_1_1.py', description='MLDS2018 hw1-1.1 Simulate Functions')
parser.add_argument('--sr', type=int, default=1000)
parser.add_argument('--rng', type=list, default=[0., 5.])
parser.add_argument('--func', type=str, default='damping')

parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='sgd')
parser.add_argument('--epoch', type=int, default=2500)
parser.add_argument('--batch', type=int, default=-1)
parser.add_argument('--model', type=str, default='shallow')

parser.add_argument('--prefix', type=str, default='unnamed')
parser.add_argument('--save_dir', type=str, default='./')
parser.add_argument('--result_dir', type=str, default='./')
parser.add_argument('--result_file', type=str, default=None)
args = parser.parse_args()


# construct stimulated function
def target_func(x):
    if args.func == 'damping': return math.exp(-x) * math.cos(2 * math.pi * x)
    elif args.func == 'triangle': return 1. if x % 2 < 1 else 0.

tr_x = np.arange(*args.rng, 1 / args.sr, dtype=np.float32)
tr_y = np.array([target_func(x) for x in tr_x], dtype=np.float32)
tr_x = tr_x.reshape(-1, 1)
tr_y = tr_y.reshape(-1, 1)


# construct model
X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])

if   args.model == 'shallow': out = shallow(X, 'shallow', 288)
elif args.model == 'medium' : out = medium(X, 'medium', 28)
elif args.model == 'deep'   : out = deep(X, 'deep', 16)

mse = tf.losses.mean_squared_error(Y, out)
if   args.opt == 'sgd' : train_step = tf.train.GradientDescentOptimizer(args.lr).minimize(mse)
elif args.opt == 'adam': train_step = tf.train.AdamOptimizer(args.lr).minimize(mse)


# start training
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    epoch_loss = []
    total_steps = max(int(np.ceil(len(tr_x) / args.batch)), 1)

    for n_epoch in range(args.epoch):
        cost = 0
        n_step = 0
        rdn_x, rdn_y = shuffle(tr_x, tr_y)

        for batch_x, batch_y in batch(rdn_x, rdn_y, size=args.batch):
            tStart = time.time()
            sess.run(train_step, feed_dict={X: batch_x, Y: batch_y})
            step_loss = sess.run(mse, feed_dict={X: batch_x, Y: batch_y})
            tEnd = time.time()
             
            if n_epoch % 100 == 99:
                stdout(n_epoch, n_step, total_steps, tEnd-tStart, step_loss)
            cost += tEnd - tStart
            n_step += 1

        epoch_loss.append(sess.run(mse, feed_dict={X: tr_x, Y: tr_y}))
        if n_epoch % 100 == 99:
            stdout(n_epoch, n_step, total_steps, cost, epoch_loss[-1])

    pred = sess.run(out, feed_dict={X: tr_x})


# save prediction and loss
lossfile = open(args.save_dir + args.func + '_' + args.model + '_loss.csv', 'w')
print('epoch,loss', file=lossfile)
for i, loss in enumerate(epoch_loss): print(i, loss, sep=',', file=lossfile)
lossfile.close()

if args.result_file: outfile = open(args.result_file, 'w')
else: outfile = open(args.result_dir + args.func + '_' + args.model + '.csv', 'w')
tr_x, pred = tr_x.squeeze(), pred.squeeze()
print('x,f(x)', file=outfile)
for i, x in enumerate(tr_x): print(x, pred[i], sep=',', file=outfile)
outfile.close()
