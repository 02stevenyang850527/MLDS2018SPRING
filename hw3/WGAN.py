
import matplotlib
matplotlib.use('Agg')  # if necessary
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import time
import sys
import os


noise_dim = 128


def conv2d_layer(input_tensor, filter_shape, strides, name, padding='SAME'):
    with tf.variable_scope(name):
        filters = tf.get_variable('filters', filter_shape, dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        tf.summary.histogram("filters", filters)
        bias = tf.get_variable('bias', [filter_shape[3]], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        output = tf.nn.conv2d(input_tensor, filters, strides, padding)
        return output


def conv2d_transpose_layer(input_tensor, filter_shape, strides, output_shape, name, padding='SAME'):
    with tf.variable_scope(name):
        filters = tf.get_variable('filters', filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        tf.summary.histogram("filters", filters)
        bias = tf.get_variable('bias', [filter_shape[2]], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        output = tf.nn.conv2d_transpose(input_tensor, filters, output_shape, strides, padding)
        return output


def maxpool2d(input_tensor, name, k=2):
    with tf.variable_scope(name):
        return tf.nn.max_pool(input_tensor, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


def dense_layer(input_tensor, input_dim, output_dim, name, activation=tf.nn.relu):
    with tf.variable_scope(name):
        W = tf.get_variable('weights', [input_dim, output_dim], dtype=tf.float32,  initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable('bias', [output_dim], dtype=tf.float32)
        output = tf.nn.bias_add(tf.matmul(input_tensor, W), b)
        return activation(output)


def lrelu(x, alpha=0.2):
    return tf.maximum(x, alpha*x)
    #return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def batch_norm(x, isTrain, name, decay=0.9, eps=1e-5, stddev=0.02):
    return tf.contrib.layers.batch_norm(inputs = x,
                                        decay = decay,
                                        updates_collections=None,
                                        epsilon = eps,
                                        scale=True,
                                        is_training = isTrain,
                                        scope = name)


def generator(noise_tensor, isTrain_tensor, reuse=None):
    with tf.variable_scope('generator', reuse=reuse):
        n_inputs = tf.shape(noise_tensor)[0]
        '''
        d1 = dense_layer(noise_tensor, noise_dim, 4*4*512, 'dense1')
        d1 = tf.reshape(d1, [-1,4,4,512])'''
        noise_tensor = tf.reshape(noise_tensor, [-1,1,1,128])
        dc0 = conv2d_transpose_layer(noise_tensor, filter_shape=[4,4,512,128], strides=[1,1,1,1],
                                     output_shape=[n_inputs, 4, 4, 512],
                                     name='deconv0',
                                     padding='VALID'
                                    )
        b1 = tf.nn.relu(batch_norm(dc0, isTrain_tensor, name='batch_norm1'))

        dc1 = conv2d_transpose_layer(b1, filter_shape=[4,4,256,512], strides=[1,2,2,1], 
                                     output_shape=[n_inputs, 8, 8, 256],
                                     name='deconv1'
                                    )
        b2 = tf.nn.relu(batch_norm(dc1, isTrain_tensor, name='batch_norm2'))
        dc2 = conv2d_transpose_layer(b2, filter_shape=[4,4,128,256], strides=[1,2,2,1], 
                                     output_shape=[n_inputs, 16, 16, 128],
                                     name='deconv2'
                                    )
        b3 = tf.nn.relu(batch_norm(dc2, isTrain_tensor, name='batch_norm3'))
        dc3 = conv2d_transpose_layer(b3, filter_shape=[4,4,64,128], strides=[1,2,2,1], 
                                     output_shape=[n_inputs, 32, 32, 64],
                                     name='deconv3'
                                    )
        b4 = tf.nn.relu(batch_norm(dc3, isTrain_tensor, name='batch_norm4'))
        gen_image = conv2d_transpose_layer(b4, filter_shape=[4,4,3,64], strides=[1,2,2,1], 
                                           output_shape=[n_inputs, 64, 64, 3],
                                           name='generated_image'
                                          )
        return tf.nn.tanh(gen_image)


def discriminator(images, isTrain_tensor, reuse=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        # image shape = [None, 64, 64, 3]
        conv1 = conv2d_layer(images, filter_shape=[4,4,3,64], strides=[1,2,2,1],
                             name='conv1',
                            )
        b1 = lrelu(batch_norm(conv1, isTrain_tensor, name='batch_norm1'))
        conv2 = conv2d_layer(b1, filter_shape=[4,4,64,128], strides=[1,2,2,1],
                             name='conv2',
                            )
        b2 = lrelu(batch_norm(conv2, isTrain_tensor, name='batch_norm2'))
        conv3 = conv2d_layer(b2, filter_shape=[4,4,128,256], strides=[1,2,2,1],
                             name='conv3',
                            )
        b3 = lrelu(batch_norm(conv3, isTrain_tensor, name='batch_norm3'))
        conv4 = conv2d_layer(b3, filter_shape=[4,4,256,512], strides=[1,2,2,1],
                             name='conv4',
                            )
        b4 = lrelu(batch_norm(conv4, isTrain_tensor, name='batch_norm4'))
        conv5 = conv2d_layer(b4, filter_shape=[4,4,512,1], strides=[1,1,1,1],
                             name='conv5',
                             padding='VALID',
                            )
        score = tf.reshape(conv5, [-1, 1])
        return score


class Dataset:

    def __init__(self, isTrain=True):
        self.isTrain = isTrain
        self.cnt = 0
        print("Loading training images...")
        self.images = np.load('extra_images.npy')
        print("Loading completed...")
        self.images = self.images/127.5 -1.0
        self.n_samples = self.images.shape[0]
        if (isTrain):
            index = np.random.permutation(self.n_samples)
            self.images = self.images[index]

    def next_batch(self, batch_size):
        if self.cnt + batch_size > self.n_samples:
            used = np.arange(self.cnt)
            np.random.shuffle(used)
            rest = np.arange(self.cnt, self.n_samples)
            shuffle = np.append(rest, used)
            self.images = self.images[shuffle,:]
            self.cnt = 0

        start = self.cnt
        self.cnt += batch_size
        end = self.cnt
        batch_x = self.images[start:end,:]

        return batch_x


def train(resume=0):

    training_data = Dataset(isTrain = True)

    noise = tf.placeholder(tf.float32, [None,noise_dim], name='noise')
    images = tf.placeholder(tf.float32, [None,64,64,3], name='image')
    isTrain = tf.placeholder(tf.bool, name='isTrain')
    print("Build generator...")
    gen_images = generator(noise, isTrain, reuse=False)
    print("Build discriminator...")
    score_real = discriminator(images, isTrain, reuse=False)
    score_fake = discriminator(gen_images, isTrain, reuse=True)

    batch_size = 64
    lam = 10.0

    d_loss =  tf.reduce_mean(score_fake - score_real)
    g_loss = tf.reduce_mean(-score_fake)

    vars_gen = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    vars_dis = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    n_epoch = 100
    total_steps = int(training_data.n_samples/batch_size)
    epoch2save = 5
    bar_length = 20
    ncritics = 5

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars

    saver = tf.train.Saver(var_list=var_list, max_to_keep=int(n_epoch/epoch2save))

    model_dir = 'save/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    log_dir = 'log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    pic_dir = 'pics/'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    d_trainer = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(d_loss, var_list=vars_dis, colocate_gradients_with_ops=True)
    g_trainer = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(g_loss, var_list=vars_gen, colocate_gradients_with_ops=True)

    clip_ops = []
    for var in vars_dis:
        clip_bounds = [-.01, .01]
        clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
    clip_disc_weights = tf.group(*clip_ops)

    tf.get_variable_scope().reuse_variables()

    tf.summary.scalar('Generator_loss', g_loss)
    tf.summary.scalar('Discriminator_loss', d_loss)

    merged = tf.summary.merge_all()

    init = tf.global_variables_initializer()


    with tf.Session() as sess:
        if resume==0:
            sess.run(init)
            print('Start Training from scratch\n')
        else:
            print('Loading Epoch:{} trained model & Keep Training...\n'.format(resume))
            saver.restore(sess, './save/model.ckpt-{}'.format(resume))

        writer = tf.summary.FileWriter(log_dir, sess.graph)

        for e in range(resume, n_epoch):
            print("Epoch:{}/{}".format(e+1, n_epoch))
            for s in range(total_steps):
                t = time.time()
                batch_x = training_data.next_batch(batch_size)
                noise_batch = np.random.normal(0, 1, (batch_size, noise_dim))
                # Train discriminator
                _ = sess.run(d_trainer, feed_dict={images: batch_x, 
                                                   noise: noise_batch, 
                                                   isTrain: True})
                _ = sess.run([clip_disc_weights])
                if (s%ncritics == ncritics-1):
                # Train generator
                    noise_batch = np.random.normal(0, 1, (batch_size, noise_dim))
                    _ = sess.run(g_trainer, feed_dict={images: batch_x,
                                                       noise: noise_batch,
                                                       isTrain: True})

                summary = sess.run(merged, feed_dict={images: batch_x,
                                                      noise: noise_batch, 
                                                      isTrain: False})
                writer.add_summary(summary, e*total_steps + s+1)

                remain_time = round((time.time()-t)*(total_steps-s))
                progress = round(bar_length*(s/total_steps))
                text = "\rProgress: [%s] - ETA: %-4ds - "%(
                      '='*(progress-1)+'>'+'.'*(bar_length-progress),
                      remain_time)
                sys.stdout.write(text)
                sys.stdout.flush()

            print('')

            #### plot figures ####
            r, c = 5, 5
            noise_in = np.random.normal(0, 1, (r * c, 128))
            predictions = sess.run(gen_images, feed_dict={noise:noise_in, isTrain:False})
            fig, axs = plt.subplots(r, c)
            cnt=0
            predictions = np.clip(predictions*255, 0, 255)
            predictions = predictions.astype(np.uint8)
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(predictions[cnt, :,:,:])
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig("./pics/output_{}.png".format(e+1))
            plt.close()
            #######################

            if (e%epoch2save==epoch2save-1):
                saver.save(sess, model_dir+'model.ckpt', global_step=e+1)



def test(resume):
    bar_length = 20
    model_dir = 'save/'
    print("Build generator...")
    noise = tf.placeholder(tf.float32, [None,noise_dim], name='noise')
    isTrain = tf.placeholder(tf.bool, name='isTrain')
    gen_images = generator(noise, isTrain)

    saver = tf.train.Saver()
    output_dir = 'samples/'
    seed = 5
    np.random.seed(seed)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    r, c = 5, 5
    with tf.Session() as sess:
        if resume == 0:
            print('Loading latest trained model & Testing...\n')
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        else:
            print('Loading Epoch:{} trained model & Testing...\n'.format(resume))
            saver.restore(sess, './save/model.ckpt-{}'.format(resume))
        noise_in = np.random.normal(0, 1, (r * c, 128))
        #noise_in = np.random.triangular(-1,0,1,(r*c, 128))
        #noise_in = (np.random.poisson(4,(r*c, 128)) - 3)/2
        #noise_in = np.random.weibull(100, (r*c, 128)) # result in mode collapse
        #noise_in = np.random.noncentral_chisquare(3, 0, (r*c, 128))/np.exp(1)
        #noise_in = np.random.lognormal(0, 0.77, (r*c, 128))/np.pi
        predictions = sess.run(gen_images, feed_dict={noise:noise_in, isTrain:False})

    #### plot figures ####
    fig, axs = plt.subplots(r, c)
    cnt=0
    predictions = np.clip(predictions*255, 0, 255)
    predictions = predictions.astype(np.uint8)
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(predictions[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(output_dir+"/gan.png")
    plt.close()
    #######################



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', help='testing mode', action='store_true')
    parser.add_argument('-r', '--resume', help='resume training', type=int, default=0)
    args = parser.parse_args()

    if args.test:
        print('Testing...')
        test(args.resume)
    else:
        print('Training...')
        train(args.resume)

