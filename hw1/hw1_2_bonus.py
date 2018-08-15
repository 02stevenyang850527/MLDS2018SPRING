from sklearn import manifold
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import sys
import os


class MyModel:

    def __init__(self, input_tensor, name, layers=None):
        self.layers = []            # Stack of layers.
        self.outputs = input_tensor
        self.name = name
        self.tensors= []

        # Add to the model any layers passed to the constructor.
        if layers:
            for layer in layers:
                self.add(layer)

    def add(self, layer):
        if len(layer) > 1:
            self.layers.append(layer[0])
            self.outputs = layer[0]
            self.tensors.append(layer[1])
            self.tensors.append(layer[2])
        else:
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

def lrelu(x, alpha=0.2):
            return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def dense_layer(input_tensor, input_dim, output_dim, name, activation=lrelu):
    with tf.variable_scope(name):
        W = tf.get_variable('weights', [input_dim, output_dim], dtype=tf.float32)
        b = tf.get_variable('bias', [output_dim], dtype=tf.float32)
        output = tf.nn.bias_add(tf.matmul(input_tensor, W), b)
        return activation(output), W, b


def test_net(x, name, num_layers=2, hidden_units=32, num_classes=10 ,reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        model = MyModel(x,name)
        for l in range(num_layers):
            dim = np.prod(model.outputs.get_shape().as_list()[1:])
            model.add(dense_layer(model.outputs, dim,hidden_units, name="fc"+str(l+1)))

        dim = np.prod(model.outputs.get_shape().as_list()[1:])
        model.add(dense_layer(model.outputs, dim, num_classes, name="output", activation=tf.identity))
        model.summary()
        return model


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

    sample_pt = 20001
    x_train = np.linspace(-4,4,sample_pt)
    y_train = np.sin(x_train)
    index = np.random.permutation(sample_pt)
    x_train = x_train[index].reshape(-1,1)
    y_train = y_train[index].reshape(-1,1)

    mydata = dataset(x_train, y_train)
    lr = 0.001
    batch_size = 128
    num_epoch = 50

    X = tf.placeholder(tf.float32, [None, 1], name='Input')
    Y = tf.placeholder(tf.float32, [None, 1], name='Output')
    model_name = 'DNN'
    num_layers = 3
    num_units = 3
    model = test_net(X,name=model_name,
                    num_layers=num_layers,
                    hidden_units=num_units,
                    num_classes=1
                   )
    pred = model.outputs
    loss_op = tf.reduce_mean(tf.squared_difference(pred, Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss_op)

    total_steps = int(sample_pt/batch_size)
    init = tf.global_variables_initializer()
    bar_length = 20
    train_loss = np.zeros(num_epoch)
    train_param = np.zeros((num_epoch,34))
    param_storage = np.zeros((num_epoch*10, 34))
    cnt = 0
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epoch):
            print("")
            print("Epoch:{}/{}".format(epoch+1, num_epoch))
            sample = np.random.permutation(total_steps-1)[0:10]
            train_loss_tmp = []
            for step in range(total_steps):
                batch_x, batch_y = mydata.next_batch(batch_size)

                t = time.time()
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                loss = sess.run(loss_op, feed_dict={X: batch_x, Y: batch_y})
                train_loss_tmp.append(loss)

                if step in sample:
                    param_tmp = []
                    for p in tf.trainable_variables():
                        p1 = sess.run(p).reshape(-1,).tolist()
                        for variable in p1:
                            param_tmp.append(variable)
                    param_storage[cnt,:] = np.array(param_tmp)
                    cnt += 1

                progress = round(bar_length*(step/total_steps))
                remain_time = round((time.time()-t)*(total_steps-step))
                text = "\rProgress: [%s] - ETA: %-4ds - loss: %-5.3f"%(
                  '='*(progress-1)+'>'+'.'*(bar_length-progress),
                  remain_time,
                  loss
                  )
                sys.stdout.write(text)
                sys.stdout.flush()

            train_loss[epoch] = np.mean(train_loss_tmp)
            train_param_tmp = []
            for p in tf.trainable_variables():
                p1 = sess.run(p).reshape(-1,).tolist()
                for variable in p1:
                    train_param_tmp.append(variable)
            
            train_param[epoch,:] = np.array(train_param_tmp)

        print('')
        print('Optimization Finished! Start predict...')
        x_train = np.linspace(-4,4,sample_pt).reshape(-1,1)
        y_train = np.sin(x_train).reshape(-1,1)
        y_pred, loss_pred = sess.run([pred, loss_op], feed_dict={X: x_train, Y:y_train})

        tsne = manifold.TSNE(n_components=2, init='pca')
        X_tsne = tsne.fit_transform(np.concatenate((param_storage, train_param)))
        print("Prediction Loss:", loss_pred)
        plt.figure()
        plt.plot(x_train, y_train, label='y=sin(x)')
        plt.plot(x_train, y_pred, label='pred')
        plt.xlabel("input x")
        plt.ylabel("y = f(x)")
        plt.legend()
        pic_dir = './pic'
        if not os.path.exists(pic_dir):
            os.makedirs(pic_dir)
        plt.savefig(pic_dir+'/y=sinx.png')
        plt.figure()
        pts = param_storage.shape[0]
        T = np.arctan2(X_tsne[:pts,0], X_tsne[:pts,1])
        plt.scatter(X_tsne[:pts,0], X_tsne[:pts,1], s=75, c=T, alpha=0.5)
        plt.plot(X_tsne[pts:,0],X_tsne[pts:,1],'k')
        plt.annotate(train_loss[0], (X_tsne[pts,0], X_tsne[pts,1]))
        plt.annotate(train_loss[-1], (X_tsne[-1,0], X_tsne[-1,1]))
        plt.savefig(pic_dir+'/error_surface.png')
        plt.show()
