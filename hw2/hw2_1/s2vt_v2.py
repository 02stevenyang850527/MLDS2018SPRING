from keras.preprocessing.text import text_to_word_sequence
from tensorflow.contrib.seq2seq import sequence_loss
import tensorflow as tf
import numpy as np
import argparse
import pickle
import time
import json
import sys
import os

word2id = pickle.load(open('word2id.pkl','rb'))
id2word = dict((v,k) for k, v in word2id.items())
vocab_size = len(word2id)
word_dim = vocab_size
nframe = 80
image_dim = 4096
enc_dim = 256
dec_dim = 256
max_seqlen = 60

class Dataset:

    def __init__(self, id_path, feat_path, label_path, shuffle):
        ids = []
        self.cnt = 0
        self.label = []
        self.label2id = []
        self.shuffle = shuffle

        with open(id_path,'r') as f:
            for line in f:
                ids.append(line[:-1])
        ids = np.array(ids)
        self.ids = ids


        print('Loading video features...')
        self.feats = dict((video_id, np.load('{}/{}.npy'.format(feat_path, video_id)) ) for video_id in self.ids)
        print('Loading completed!')


        if self.shuffle: # only training mode will shuffle
            with open(label_path, 'r') as f:
                info = json.load(f)
            for video in info:
                for caption in video['caption']:
                    self.label.append(caption)
                    self.label2id.append(video['id'])

            self.label = np.array(self.label)
            self.label2id = np.array(self.label2id)
            self.n_samples = len(self.label2id)
            index = np.random.permutation(self.n_samples)
            self.label = self.label[index]
            self.label2id = self.label2id[index]
        else:
            self.n_samples = len(self.ids)

    def next_batch_for_test(self, batch_size):
        start = self.cnt
        if self.cnt == self.n_samples:
            print('Already iterate all data')
            return False, False, False, False
        if self.cnt + batch_size >= self.n_samples:
            batch_ids = self.ids[self.cnt:]
            self.cnt = self.n_samples
        else:
            self.cnt += batch_size
            end = self.cnt
            batch_ids = self.ids[start:end]

        size = len(batch_ids)
        batch_x = np.zeros((size, nframe, image_dim))
        batch_y = np.zeros((size, max_seqlen))
        batch_seqlen = np.zeros((size, max_seqlen))
        for i, index in enumerate(batch_ids):
            batch_x[i,:] = self.feats[index]

        return batch_x, batch_ids

    def next_batch(self, batch_size):
        if self.cnt + batch_size > self.n_samples:
            used = np.arange(self.cnt)
            np.random.shuffle(used)
            rest = np.arange(self.cnt, self.n_samples)
            shuffle = np.append(rest, used)
            self.label = self.label[shuffle]
            self.label2id = self.label2id[shuffle]
            self.cnt = 0

        start = self.cnt
        self.cnt += batch_size
        end = self.cnt
        batch_ids = self.label2id[start:end]
        batch_caption = self.label[start:end]

        batch_x = np.zeros((batch_size, nframe, image_dim))
        batch_y = np.zeros((batch_size, max_seqlen))
        batch_seqlen = np.zeros((batch_size, max_seqlen))

        for i, index in enumerate(batch_ids):
            batch_x[i,:] = self.feats[index]
            vector, seqlen= self.word2vec(batch_caption[i])
            batch_y[i,:] = np.concatenate((vector, np.zeros((max_seqlen - seqlen))))
            batch_seqlen[i] = np.concatenate((np.ones(seqlen), np.zeros((max_seqlen-seqlen))))

        return batch_x, batch_y, batch_seqlen


    def word2vec(self, sentence):
        # <PAD>: 0
        sen = text_to_word_sequence(sentence)
        sen = sen + ['<EOS>']
        vector = [word2id[word] if word in word2id.keys() else word2id['<UNK>']for word in sen]
        seq_len = len(sen)
        return np.array(vector), seq_len


def schedule_sampling(batch_size, max_len, prob):
    sampling = np.zeros((batch_size, max_len), dtype = bool)
    for b in range(batch_size):
        for l in range(max_len):
            if np.random.uniform(0,1,1) < prob:
                sampling[b,l] = True
    return sampling


class my_model:
    def __init__(self, isTrain):
        self.isTrain = isTrain

    def build_model(self):
        video = tf.placeholder(tf.float32, [None, nframe, image_dim], name='video')
        caption = tf.placeholder(tf.int32, [None, max_seqlen], name='caption')
        caption_len = tf.placeholder(tf.float32, [None, max_seqlen], name='caption_length')
        keep_prob = tf.placeholder(tf.float32, name='Dropout_rate')
        sampling = tf.placeholder(tf.bool, [None, max_seqlen], name='sampling_prob')
        # Word embeddings
        if word_dim == vocab_size:
            print('Method: use one-hot embedding matrix\n')
            embeddings = tf.constant(np.identity(word_dim), name='Word_Embeddings', dtype=tf.float32)
        else:
            print('Method: train an embedding matrix\n')
            embeddings = tf.Variable(tf.truncated_normal([vocab_size, word_dim], 0.0, 1.0/word_dim), dtype=tf.float32, name='Word_Embeddings')

        if self.isTrain:
            video_with_noise = tf.add(video, tf.abs(tf.random_normal(shape=tf.shape(video), mean=0.0, stddev=0.01, dtype=tf.float32)))
            enc_input = tf.transpose(video_with_noise, [1, 0, 2]) # Turn to time major
        else:
            enc_input = tf.transpose(video, [1, 0, 2]) # Turn to time major

        with tf.variable_scope('Encoder'):
            encoder = tf.contrib.rnn.LSTMCell(enc_dim)
            encoder = tf.contrib.rnn.DropoutWrapper(encoder, output_keep_prob=keep_prob)

        with tf.variable_scope('Decoder'):
            decoder = tf.contrib.rnn.LSTMCell(dec_dim)
            decoder = tf.contrib.rnn.DropoutWrapper(decoder, output_keep_prob=keep_prob)

        n_input = tf.shape(video)[0]
        enc_initial_state = tf.zeros([n_input, enc_dim])
        dec_initial_state = tf.zeros([n_input, dec_dim])
        state1 = (tf.zeros_like(enc_initial_state), tf.zeros_like(enc_initial_state))
        state2 = (tf.zeros_like(dec_initial_state), tf.zeros_like(dec_initial_state))

        #zero_padding = tf.zeros([n_input, word_dim])
        #word_padding = tf.zeros_like(zero_padding)
        word_padding = tf.nn.embedding_lookup(embeddings, tf.zeros([n_input,], tf.int32))  # <PAD>: 0
        # Encoding
        for i in range(nframe):
            with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
                output1, state1 = encoder(enc_input[i], state1)
            with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE):
                output2, state2 = decoder(tf.concat([word_padding, output1], axis=1), state2)
            
        # Decoding
        sample = tf.transpose(sampling, [1, 0])
        cap = tf.transpose(caption, [1, 0])

        with tf.variable_scope('Output'):
            Wo = tf.Variable(tf.truncated_normal([dec_dim, vocab_size], 0, 1/dec_dim), name='W')
            bo = tf.Variable(tf.zeros([vocab_size]), name='b')
        
        loss = 0
        prediction = []
        logit_sequence = []

        for i in range(max_seqlen):
            if i==0:
                current_embeddings = tf.nn.embedding_lookup(embeddings, tf.ones([n_input,], tf.int32)) # <BOS>: 1
            else:
                if self.isTrain:
                    current_embeddings = tf.nn.embedding_lookup(embeddings, tf.where(sample[i-1], cap[i-1], predict))
                    # Take previous output or label as input
                else:
                    current_embeddings = tf.nn.embedding_lookup(embeddings, predict)

            with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
                output1, state1 = encoder(tf.zeros_like(enc_input[0]), state1)
            with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE):
                output2, state2 = decoder(tf.concat([current_embeddings, output1], axis=1), state2)

            logit = tf.nn.bias_add(tf.matmul(output2, Wo), bo)
            logit_sequence.append(logit)
            predict = tf.argmax(logit, axis=1)
            predict = tf.cast(predict, tf.int32)
            prediction.append(predict)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=tf.one_hot(cap[i], depth=vocab_size))
            loss += tf.reduce_sum(tf.multiply(cross_entropy, caption_len[:,i]))

        loss = tf.divide(loss, tf.reduce_sum(caption_len))
        #loss = sequence_loss(tf.stack(logit_sequence),caption, caption_len)
        prediction = tf.transpose(tf.stack(prediction), [1, 0])
        return video, caption, caption_len, sampling, keep_prob, prediction, loss


def train(resume):
    id_path = 'training_data/id.txt'
    feat_path = 'training_data/feat/'
    label_path = 'training_label.json'
    S2VT = my_model(isTrain=True)
    video, caption, caption_len, sampling, keep_prob, prediction, loss = S2VT.build_model()
    mydata = Dataset(id_path, feat_path, label_path, shuffle=True)

    lr = 0.001
    n_epoch = 200
    batch_size = 256
    total_steps = int(mydata.n_samples/batch_size)
    bar_length = 20
    model2save = 5

    opt = tf.train.AdamOptimizer(lr)
    gvs = opt.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in gvs]
    train_op = opt.apply_gradients(capped_gvs)

    saver = tf.train.Saver(max_to_keep=int(n_epoch/model2save))
    model_dir = 'save/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    prob = 1.0

    init = tf.global_variables_initializer()
    loss_logger = open('24232_s2vt_loss.txt','w')
    print('Open loss logger')
    with tf.Session() as sess:
        if resume==0:
            sess.run(init)
            print('Start Training from scratch\n')
        else:
            print('Loading Epoch:{} trained model & Keep Training...\n'.format(resume))
            saver.restore(sess, './save/model.ckpt-{}'.format(resume))
            prob = 0.7

        for e in range(resume, n_epoch):
            print("Epoch:{}/{}".format(e+1, n_epoch))
            #prob = 1/(1+np.exp((e-500)/200))
            prob -= 0.003
            #prob = 0.8
            loss_tmp = []
            for s in range(total_steps):
                t = time.time()
                batch_x, batch_y, batch_seqlen = mydata.next_batch(batch_size)
                training_loss, predict, _ = sess.run([loss, prediction, train_op], feed_dict={
                                                      video: batch_x, caption: batch_y,
                                                      sampling: schedule_sampling(batch_size, max_seqlen, prob),
                                                      caption_len: batch_seqlen,
                                                      keep_prob: 0.5
                                                    })
                remain_time = round((time.time()-t)*(total_steps-s))
                loss_tmp.append(training_loss)

                progress = round(bar_length*(s/total_steps))
                text = "\rProgress: [%s] - ETA: %-4ds - loss: %-5.3f "%(
                      '='*(progress-1)+'>'+'.'*(bar_length-progress),
                      remain_time,
                      training_loss
                      )
                sys.stdout.write(text)
                sys.stdout.flush()
            print('')
            print('Samping the last batch from Epoch: {}'.format(e+1))
            print(np.mean(loss_tmp), file=loss_logger)

            for i in range(4):
                c = predict[i]
                find_eos = np.where(c == word2id['<EOS>'])[0]
                end_of_sentence = find_eos[0] if len(find_eos) != 0 else len(c)
                pred_caption = ' '.join([id2word[wid] for wid in c[:end_of_sentence]])
                print('Sample ',i+1)
                find_ref_eos = np.where(batch_y[i] == word2id['<EOS>'])[0]
                eos_ref = find_ref_eos[0] if len(find_ref_eos) != 0 else len(batch_y[i])
                print('Reference :' + ' '.join([id2word[wid] for wid in batch_y[i,:eos_ref]]))
                print('Prediction:' + pred_caption)
            if (e%model2save == model2save-1):
                saver.save(sess, model_dir+'model.ckpt', global_step=e+1)

        loss_logger.close()


def test(resume, path, of):
    id_path = path+'/id.txt'
    feat_path = path+'/feat/'
    label_path = ''
    mydata = Dataset(id_path, feat_path, label_path, shuffle=False)
    S2VT = my_model(isTrain=False)
    video, caption, caption_len, sampling, keep_prob, prediction, loss = S2VT.build_model()

    batch_size = 1
    total_steps = int(mydata.n_samples/batch_size)
    bar_length = 20
    saver = tf.train.Saver()
    model_dir = 'save/'

    with tf.Session() as sess:
        if resume == 0:
            print('Loading latest trained model & Testing...\n')
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        else:
            print('Loading Epoch:{} trained model & Testing...\n'.format(resume))
            saver.restore(sess, './save/model.ckpt-{}'.format(resume))

        output_path = of
        of = open(output_path,'w')
        out = ''

        for s in range(total_steps):
            batch_x, batch_ids = mydata.next_batch_for_test(batch_size)
            t = time.time()
            predict = sess.run(prediction, feed_dict={
                                              video: batch_x,
                                              keep_prob: 1.0
                                            })
            remain_time = round((time.time()-t)*(total_steps-s))
            progress = round(bar_length*(s/total_steps))
            text = "\rProgress: [%s] - ETA: %-4ds "%(
                  '='*(progress-1)+'>'+'.'*(bar_length-progress),
                  remain_time,
                  )
            sys.stdout.write(text)
            sys.stdout.flush()

            
            for index, vid in enumerate(batch_ids):
                c = predict[index]
                find_eos = np.where(c == word2id['<EOS>'])[0]
                end_of_sentence = find_eos[0] if len(find_eos) != 0 else len(c)
                if end_of_sentence > len(c):
                    end_of_sequence = len(c)
                #pred_caption = ' '.join([id2word[wid] for wid in c[:end_of_sentence]])
                
                pred_caption = ''
                for k, wid in enumerate(c[:end_of_sentence]):
                    if k == 0:
                        pred_caption += id2word[wid]
                    else:
                        if wid != c[k-1]:
                            pred_caption += ' ' + id2word[wid]
                
                out += vid + ',' + pred_caption + '\n'
            
        print('')
        out = out[:-1]
        of.write(out)
        of.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', help='testing mode', action='store_true')
    parser.add_argument('-r', '--resume', help='resume training', type=int, default=0)
    parser.add_argument('-p', '--path', help='path of testing data', type=str, default='')
    parser.add_argument('-o', '--output', help='path of prediction output', type=str, default='')
    args = parser.parse_args()

    if args.test:
        print('Testing...')
        test(args.resume, args.path, args.output)
    else:
        print('Training...')
        train(args.resume)

