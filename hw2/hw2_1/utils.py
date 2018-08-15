import sys
import json
import random
import numpy as np
import tensorflow as tf


def _shuffle(*args):
    for arg in args: assert len(arg) == len(args[0])

    idx = np.arange(len(args[0]))
    np.random.shuffle(idx)

    if len(args) == 1: return args[0][idx]
    else: return (arg[idx] for arg in args)


def _batch(*args, size=None):
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


def read_data(feat_dir, label_path, id_path):
    ids = np.array([ name[:-1] for name in open(id_path, 'r') ])
    feats = np.array([ np.load('{}/{}.npy'.format(feat_dir, ids[i])) for i in range(len(ids)) ])
    
    infos = json.load(open(label_path, 'r'))
    labels = { infos[i]['id']: infos[i]['caption'] for i in range(len(infos)) }
    return feats, labels, ids


class word2vec:
    def __init__(self, corpus_list, min_cnt=3, del_chars=' .\n', use_UNK=False):
        self.word_dict = {}
        self.use_UNK = use_UNK

        for corpus in corpus_list:
            for sentence in corpus:
                words = sentence.strip(del_chars).split()
                for word in words:
                    if word in self.word_dict: self.word_dict[word] += 1
                    else: self.word_dict[word] = 1

        self.dataset = {'<BOS>': 0, '<EOS>': 1}
        if use_UNK: self.dataset['<UNK>'] = len(self.dataset)
        
        for word, cnt in self.word_dict.items():
            if cnt >= min_cnt: self.dataset[word] = len(self.dataset)

    def extend(self, corpus_list, del_chars=' .\n'):
        for corpus in corpus_list:
            for sentence in corpus:
                words = sentence.strip(del_chars).split()
                for word in words:
                    if word in word_dict: 
                        self.word_dict[word] += 1
                        if self.word_dict[word] == min_cnt:
                            self.dataset[word] = len(self.dataset)
                    else: self.word_dict[word] = 1
    
    def sen2vec(self, sentence, del_chars=' .\n'):
        words  = ['<BOS>']
        words += sentence.strip(del_chars).split()
        words += ['<EOS>']

        idx = self._word2idx(words)
        sentence_size = len(idx)
        dataset_size  = len(self.dataset)
        
        vector_sequence = np.zeros((sentence_size, dataset_size))
        vector_sequence[np.arange(sentence_size), idx] = 1
        return vector_sequence.astype(np.float32)

    def _word2idx(self, words):
        idx = []
        for word in words:
            if word in self.dataset: 
                idx.append(self.dataset[word])
            elif self.use_UNK: 
                idx.append(self.dataset['<UNK>'])
        return np.asarray(idx)


class data_generator:
    def __init__(self, feat_dir, label_path, id_path, min_cnt=3, del_chars=' .\n', use_UNK=False):
        self.feats, self.labels, self.ids = read_data(feat_dir, label_path, id_path)
        
        corpus = []
        for _, sentences in self.labels.items(): corpus += sentences

        self.word2vec = word2vec([corpus], min_cnt, del_chars, use_UNK)

    def flow(self, batch_size=None, shuffle=True):
        if batch_size == None or batch_size == -1: batch_size = len(self.feats)
        if shuffle: self.feats, self.ids = _shuffle(self.feats, self.ids)

        for batch_x, batch_id in _batch(self.feats, self.ids, size=batch_size):
            batch_y = [ random.choice(self.labels[item]) for item in batch_id ]
            batch_y = [ self.word2vec.sen2vec(y) for y in batch_y ]
            # pad batch_y
            # batch_y = np.array([ self.word2vec.sen2vec(y) for y in batch_y ])

            yield batch_x, batch_y

    def __len__(self):
        return len(self.feats)



if __name__ == '__main__':
    feat_dir = '../data/training_data/feat/'
    label_path = '../data/training_label.json' 
    id_path = '../data/training_id.txt'
    
    tr_gen = data_generator(feat_dir, label_path, id_path, use_UNK=True)
    print(len(tr_gen))
    for i, (batch_x, batch_y) in enumerate(tr_gen.flow(batch_size=10)):
        print('finish batch {}'.format(i+1))
        '''do something'''
