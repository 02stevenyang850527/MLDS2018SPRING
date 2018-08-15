import json
import pickle
from keras.preprocessing.text import Tokenizer, text_to_word_sequence


corpus_dir = 'training_label.json'
corpus = json.load(open(corpus_dir, 'r'))

nb_word = 6000
tokenizer = Tokenizer(num_words = nb_word)

train_txt = []

for info in corpus:
    train_txt += info['caption']

tokenizer.fit_on_texts(train_txt)

min_cnt = 3
word2id = {'<PAD>':0, '<BOS>':1, '<EOS>':2, '<UNK>': 3}

for k in tokenizer.word_counts.keys():
    if tokenizer.word_counts[k] > min_cnt:
        word = k.lower()
        new_value = len(word2id)
        word2id[word] = new_value

pickle.dump(word2id, open('word2id.pkl','wb'))
print('# of words', len(word2id)-4)
'''
print(train_txt[0])
seq = text_to_word_sequence(train_txt[0])
seq = ['<BOS>'] + seq + ['<EOS>']
print(seq)
id2word = dict((v,k) for k, v in word2id.items())
id_seq = [word2id[word] for word in seq]
print(id_seq)
new_seq = [id2word[word_id] for word_id in id_seq]
print(new_seq)
'''
