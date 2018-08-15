from tensorflow.examples.tutorials.mnist import input_data
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import sys
import os





if __name__=='__main__':

    mnist = input_data.read_data_sets("./MNIST", one_hot=True)

    Wmatrix = []
    loss_set = []
    batch_size = 128
    num_epoch = 30
    num_event = 8
    model2save = 3
    bar_length = 20
    total_steps = int(mnist.train.num_examples/batch_size)
    isFull = True

    for _ in  range(num_event):
        event_loss = []
        for step in range(num_epoch):
            tf.reset_default_graph()

            if (step % model2save == 0):
                with tf.Session() as sess:
                    saver = tf.train.import_meta_graph("save/event{}_model.ckpt-{}.meta".format(_,step))
                    saver.restore(sess,"save/event{}_model.ckpt-{}".format(_,step))
                    print("")
                    print("Model_step{} of Event {} Restore!".format(step, _))

                    graph = tf.get_default_graph()
                    X = graph.get_tensor_by_name("Input_image:0")
                    Y = graph.get_tensor_by_name("Output_label:0")
                    logits = graph.get_tensor_by_name("DNN/output/Identity:0")
                    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                             logits=logits, labels=Y))

                    temp_loss = []
                    for step in range(total_steps):
                        batch_x, batch_y = mnist.train.next_batch(batch_size)
                        # Run optimization op (backprop)
                        t = time.time()
                        loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
                        remain_time = round((time.time()-t)*(total_steps-step))

                        progress = round(bar_length*(step/total_steps))
                        text = "\rProgress: [%s] - ETA: %-4ds - loss: %-5.3f"%(
                          '='*(progress-1)+'>'+'.'*(bar_length-progress),
                          remain_time,
                          loss
                          )
                        temp_loss.append(loss)

                        sys.stdout.write(text)
                        sys.stdout.flush()

                    event_loss.append(np.mean(temp_loss))
                    if (isFull):
                        param_tmp = np.array([])
                        for p in tf.trainable_variables():
                            param = sess.run(p)
                            param_tmp = np.append(param_tmp, param[0].reshape(-1))

                    else:
                        weight = sess.run(tf.trainable_variables(scope="DNN/fc1/weights:0"))
                        bias = sess.run(tf.trainable_variables(scope="DNN/fc1/bias:0"))
                        param_tmp = np.append(weight[0].reshape(-1), bias[0].reshape(-1))

                    Wmatrix.append(param_tmp)

        loss_set.append(event_loss)
    
    print("")
    pca = PCA(n_components=2)
    wReduced = pca.fit_transform(np.array(Wmatrix))
    plt.figure()
    dots = int(num_epoch/model2save)
    for i in range(num_event):
        eventX = wReduced[i*dots:(i+1)*dots,0]
        eventY = wReduced[i*dots:(i+1)*dots,1]
        plt.scatter(eventX, eventY)
        for j in range(dots):
            plt.annotate(loss_set[i][j], (eventX[j], eventY[j]))

    pic_dir = './pic'
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    if (isFull):
        plt.savefig(pic_dir+'/vp_full.png')
    else:
        plt.savefig(pic_dir+'/vp_1layer.png')

    plt.show()
