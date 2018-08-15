import sys
import numpy as np
import tensorflow as tf


def shuffle(*args):
    for arg in args: assert len(arg) == len(args[0])

    idx = np.arange(len(args[0]))
    np.random.shuffle(idx)

    if len(args) == 1: return args[0][idx]
    else: return (arg[idx] for arg in args)


def batch(*args, size=None):
    for arg in args: assert len(arg) == len(args[0])
    
    if size == None or size == -1: size = len(args[0])
    for i in range(0, len(args[0]), size):
        if len(args) == 1: yield args[0][i:i+size]
        else: yield (arg[i:i+size] for arg in args)


def tfmt(s):
    m = s // 60
    s = s %  60

    h = m // 60
    m = m %  60

    if h != 0: return '{:.0f} h {:.0f} m {:.0f} s'.format(h, m, s)
    elif m != 0: return '{:.0f} m {:.0f} s'.format(m, s)
    else: return '{:.0f} s'.format(s)


def stdout(n_epoch, n_step, total_steps, cost, loss, **kwargs):
    line = '\rEpoch {:<5d} '.format(n_epoch+1)

    if n_step == total_steps:
        line += '[{:s}]'.format('=' * 30)
        line += ' - cost: {:4.0f} s'.format(round(cost))

    else:
        eta = round(cost * (total_steps - n_step))
        progress = int(30 * (n_step / total_steps))
        line += '[{:.<30s}]'.format('=' * progress + '>')
        line += ' - ETA: {:4.0f} s'.format(eta)
    
    line += ' - loss: {:5.4f}'.format(loss)

    for key, val in kwargs.items(): 
        line += ' - {:s}: {:5.4f}'.format(key, val)
    
    sys.stdout.write(line)
    sys.stdout.flush()
    if n_step == total_steps: print('')


def print_row(fields, lengths):
    assert len(fields) == len(lengths)

    line = ''   
    
    for i in range(len(fields)):
        line += str(fields[i])
        line += ' ' * max(1, lengths[i] - len(str(fields[i])))
    print(line)


def summary(name):
    lengths = [30, 25, 10]
    to_display = ['Layer (type)', 'Output Shape', 'Param #']

    fields_list = []
    total_cnt     = 0
    trainable_cnt = 0
    trainable_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name):
        output_shape = var.get_shape().as_list()
        param_cnt = np.prod(output_shape)
        total_cnt += param_cnt
        if var in trainable_list: trainable_cnt += param_cnt
        fields_list.append([var.name, output_shape, param_cnt])
    
    for fields in fields_list:
        lengths = [max(lengths[i], len(str(fields[i])) + 3) for i in range(len(lengths))]
    line_length = sum(lengths)
    
    print('_' * line_length)
    print_row(to_display, lengths)
    print('=' * line_length)

    for fields in fields_list: 
        print_row(fields, lengths)
    
    print('='*line_length)
    print('Total parameters:', total_cnt)
    print('Trainable parameters:', trainable_cnt)
    print('Non-trainable parameters:', total_cnt - trainable_cnt)
    print('')



if __name__ == '__main__':
    a = np.arange(10)
    b = a + 3

    a1 = shuffle(a)
    print(a1)

    a2, b2 = shuffle(a, b)
    print(a2, b2)

    for batch_a in batch(a, size=3):
        print(batch_a)

    for batch_a, batch_b in batch(a, b, size=3):
        print(batch_a, batch_b)
   

    f = 6582.36598
    print(tfmt(f))
