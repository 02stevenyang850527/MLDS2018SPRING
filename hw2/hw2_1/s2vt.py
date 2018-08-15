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
nframe = 80
image_dim = 4096
enc_dim = 256
dec_dim = enc_dim
max_seqlen = 60
total_len = nframe + max_seqlen

class Dataset:

    def __init__(self, id_path, feat_path, label_path):
        self.ids = []
        self.cnt = 0

        with open(id_path,'r') as f:
            for line in f:
                self.ids.append(line[:-1])
        self.ids = np.array(self.ids)

        with open(label_path, 'r') as f:
            info = json.load(f)
        self.label = dict((video['id'], video['caption']) for video in info)
        print('Loading video features...')
        self.feats = dict((video_id, np.load('{}/{}.npy'.format(feat_path, video_id)) ) for video_id in self.ids)
        print('Loading completed!')
        self.n_samples = len(self.ids)
        index = np.random.permutation(self.n_samples)
        self.ids = self.ids[index]

    def next_batch(self, batch_size):
        if self.cnt + batch_size > self.n_samples:
            used = np.arange(self.cnt)
            np.random.shuffle(used)
            rest = np.arange(self.cnt, self.n_samples)
            shuffle = np.append(rest, used)
            self.ids = self.ids[shuffle]
            self.cnt = 0

        start = self.cnt
        self.cnt += batch_size
        end = self.cnt
        batch_ids = self.ids[start:end]
        batch_x = np.zeros((batch_size, nframe, image_dim))
        batch_y = np.zeros((batch_size, max_seqlen))
        batch_seqlen = np.zeros((batch_size))
        for i, index in enumerate(batch_ids):
            batch_x[i,:] = self.feats[index]
            vector, seqlen= self.word2vec(self.label[index][np.random.permutation(len(self.label[index]))[0]])
            batch_y[i,:] = np.concatenate((vector, np.zeros((max_seqlen - vector.shape[0]))))
            batch_seqlen[i] = seqlen

        return batch_x, batch_y, batch_seqlen


    def word2vec(self, sentence):
        # <EOS>: 1
        # <PAD>: 0
        sen = text_to_word_sequence(sentence)
        sen = ['<BOS>'] + sen + ['<EOS>']
        vector = [word2id[word] if word in word2id.keys() else word2id['<UNK>']for word in sen]
        seq_len = len(sen)-1
        return np.array(vector), seq_len


def schedule_sampling(batch_size, max_len, prob):
    sampling = np.zeros((batch_size, max_len), dtype = bool)
    for b in range(batch_size):
        for l in range(max_len):
            if np.random.uniform(0,1,1) < prob:
                sampling[b,l] = True
    sampling[:,0] = True # To ensure <BOS> would 100% be selected.
    return sampling


class my_model:
    def __init__(self, isTrain):
        self.isTrain = isTrain

    def build_model(self):
        embeddings = tf.Variable(initial_value=np.identity(vocab_size), dtype=tf.float32, name='Word_Embeddings')
        video = tf.placeholder(tf.float32, [None, nframe, image_dim], name='Video')
        caption = tf.placeholder(tf.int32, [None, max_seqlen], name='Reference_Caption')
        sampling = tf.placeholder(tf.bool, [None, max_seqlen], name='Sampling_prob')

        # Encoding stage
        n_input = tf.shape(video)[0] # ==Batch_size

        video_padding = tf.zeros([n_input, max_seqlen, image_dim])
        enc_input = tf.concat([video, video_padding], axis=1)
        enc_input = tf.transpose(enc_input, perm=[1, 0, 2]) # Turn batch-major into time-major

        with tf.variable_scope('Encoder'):
            encoder = tf.contrib.rnn.LSTMCell(enc_dim)
            seq_len = tf.fill(tf.expand_dims(n_input, 0), tf.constant(total_len, dtype=tf.int32))
            enc_output, enc_state = tf.nn.dynamic_rnn(encoder, enc_input,
                                                      time_major = True,
                                                      sequence_length=seq_len,
                                                      dtype=tf.float32)
        # Decoding stage

        W = tf.Variable(tf.truncated_normal([dec_dim, vocab_size], 0, 1/dec_dim), name='Output_weights')
        b = tf.Variable(tf.zeros([vocab_size]), name='Output_bias')

        # Since we need to control the index of tensor in raw_rnn, turn tensor into TensorArray
        captions_ta = tf.TensorArray(dtype=tf.int32, size=max_seqlen)
        captions_ta = captions_ta.unstack(tf.transpose(caption))

        enc_output_ta = tf.TensorArray(dtype=tf.float32, size=total_len)
        enc_output_ta = enc_output_ta.unstack(enc_output)

        #sampling_ta = tf.TensorArray(dtype=tf.bool, size=max_seqlen)
        #sampling_ta = sampling_ta.unstack(tf.transpose(sampling))

        with tf.variable_scope('Decoder'):
            decoder = tf.contrib.rnn.LSTMCell(dec_dim)

        def loop_fn(time, cell_output, cell_state, loop_state):
            emit_output = cell_output  # == None for time == 0
            if cell_output is None:  # time == 0
                next_cell_state = decoder.zero_state(n_input, tf.float32)
                logits = tf.matmul(tf.ones([n_input, dec_dim]), W) + b
            else:
                next_cell_state = cell_state
                logits = tf.matmul(cell_output, W) + b
            def decoding_word_input():  # Decoding stage
                def feed_pad():
                    padding = tf.nn.embedding_lookup(embeddings, tf.zeros([n_input], dtype=tf.int32))
                    return padding
                def feed_word():
                    if self.isTrain: 
                        dec_word_input = tf.nn.embedding_lookup(embeddings, captions_ta.read(time-nframe-1))
                    else: 
                        if cell_output is None:
                            logits = tf.matmul(tf.ones([n_input, dec_dim]), W) + b
                        else:
                            logits = tf.matmul(cell_output, W) + b
                        predict = tf.argmax(logits, axis=1)
                        dec_word_input = tf.cond( tf.equal(time, nframe),
                                lambda:tf.nn.embedding_lookup(embeddings, tf.constant(word2id['<BOS>'], shape=[n_input], dtype=tf.int32)),
                                lambda:tf.nn.embedding_lookup(embeddings, predict)
                                )
                    return dec_word_input

                is_begin = (time <= nframe)
                dec_word_input = tf.cond(is_begin, feed_pad, feed_word)
                return dec_word_input

            def encoding_word_input():
                return tf.zeros([n_input, vocab_size], dtype=tf.float32) # padding

            dec_word_input = tf.cond(time < nframe, encoding_word_input, decoding_word_input)
            next_input = tf.cond(tf.equal(time, total_len),
                                 lambda: tf.zeros([n_input, enc_dim + vocab_size], dtype=tf.float32),
                                 lambda: tf.concat([enc_output_ta.read(time), dec_word_input], 1))
            elements_finished = tf.reduce_all(time >= total_len)

            return (elements_finished, next_input, next_cell_state,
                    emit_output, loop_state)


        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder, loop_fn)

        dec_output = decoder_outputs_ta.stack()
        dec_output = tf.reshape(dec_output, (-1, dec_dim))
        logits_flat = tf.add(tf.matmul(dec_output, W), b)
        logits = tf.reshape(logits_flat, (total_len, n_input, -1))
        logits = tf.transpose(logits, perm=[1, 0, 2])
        logits = logits[:, nframe:, :] # Remove <BOS>

        return video, caption, sampling, logits

def train():
    id_path = 'training_id.txt'
    feat_path = 'training_data/feat/'
    label_path = 'training_label.json'
    S2VT = my_model(isTrain=True)
    video, caption, sampling, logits = S2VT.build_model()
    mydata = Dataset(id_path, feat_path, label_path)
    prediction = tf.nn.softmax(logits, axis=-1, name='prediction')

    caption_len = tf.placeholder(tf.int32, [None], name='caption_lens')
    caption_mask = tf.sequence_mask(caption_len, max_seqlen, dtype=tf.float32)

    loss = sequence_loss(logits, caption, caption_mask)
    lr = 0.001
    n_epoch = 300
    batch_size = 16
    total_steps = int(mydata.n_samples/batch_size)
    bar_length = 20
    model2save = 5

    opt = tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(max_to_keep=int(n_epoch/model2save))
    model_dir = 'save/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    prob = 1.0

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for e in range(n_epoch):
            print("Epoch:{}/{}".format(e+1, n_epoch))
            for s in range(total_steps):
                batch_x, batch_y, batch_seqlen = mydata.next_batch(batch_size)
                t = time.time()
                training_loss, predict, _ = sess.run([loss, prediction, opt], 
                                                     feed_dict={video: batch_x, caption: batch_y,
                                                     sampling: schedule_sampling(batch_size, max_seqlen, prob),
                                                     caption_len: batch_seqlen
                                                    })
                remain_time = round((time.time()-t)*(total_steps-s))
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
            predict = np.reshape(predict, (-1, max_seqlen, vocab_size))
            word_ids = np.argmax(predict, 2)
            for i in range(4):
                c = word_ids[i]
                find_eos = np.where(c == word2id['<EOS>'])[0]
                end_of_sentence = find_eos[0] if len(find_eos) != 0 else len(c)
                pred_caption = ' '.join([id2word[wid] for wid in c[:end_of_sentence]])
                print('Sample ',i+1)
                find_ref_eos = np.where(batch_y[i] == word2id['<EOS>'])[0]
                eos_ref = find_ref_eos[0] if len(find_ref_eos) != 0 else len(batch_y[i])
                print('Reference :' + ' '.join([id2word[wid] for wid in batch_y[i,1:eos_ref]]))
                print('Prediction:' + pred_caption)
            if (e%model2save == model2save-1):
                saver.save(sess, model_dir+'model.ckpt', global_step=e)


def test():
    return 0


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', help='test mode', action='store_true')
    parser.add_argument('-k', '--keep_training', help='data directory', type=bool, default=False)
    args = parser.parse_args()

    if args.test:
        test()
        print('test')
    else:
        train()
        print('training')
