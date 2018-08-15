import os
import sys
import numpy as np
import torchvision
from PIL import Image

import torch

USE_CUDA = torch.cuda.is_available()

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


def stdout(n_epoch, n_step, total_steps, cost, loss_D, loss_G, **kwargs):
    line = '\rEpoch {:<5d} '.format(n_epoch+1)

    if n_step == total_steps:
        line += '[{:s}]'.format('=' * 30)
        line += ' - cost: {:4.0f} s'.format(round(cost))

    else:
        eta = round(cost * (total_steps - n_step))
        progress = int(30 * (n_step / total_steps))
        line += '[{:.<30s}]'.format('=' * progress + '>')
        line += ' - ETA: {:4.0f} s'.format(eta)
    
    line += ' - loss_D: {:5.4f}'.format(loss_D)
    line += ' - loss_G: {:5.4f}'.format(loss_G)

    for key, val in kwargs.items(): 
        line += ' - {:s}: {:5.4f}'.format(key, val)
    
    sys.stdout.write(line)
    sys.stdout.flush()
    if n_step == total_steps: print('')


def resize(jpgfile):
    if jpgfile.size != (64, 64): jpgfile.thumbnail((64, 64), Image.ANTIALIAS)
    return jpgfile

def load_imgs(dirname):
    img_list = [ os.path.join(dirname, filename) for filename in sorted(os.listdir(dirname), key=lambda img:int(img[0:-4])) ]
    imgs = [ np.array(resize(Image.open(filename)), dtype=np.float32) for filename in img_list ]
    imgs = np.stack(imgs)
    
    #flip_imgs = np.fliplr(imgs)
    #imgs = np.concatenate((imgs, flip_imgs), axis=0)

    imgs = imgs / 255.
    #imgs = torch.FloatTensor(imgs).permute(0, 3, 1, 2)
    #for name in img_list: print(name)
    return imgs

def load_anime_tags(filename):
    tag_file = open(filename, 'r')

    idx = []
    tags = []
    for line in tag_file:
        line = line.strip()
        infos = line.split(',')
        all_tags = [ tag.split(':')[0].strip() for tag in infos[1].split('\t') ]
        eyes_tags = [ tag for tag in all_tags if tag.endswith('eyes') and not tag.startswith('11') 
                      and not tag.startswith('bicolored') and not tag.startswith('gray') ]
        hair_tags = [ tag for tag in all_tags if tag.endswith('hair') and not tag.startswith('long')
                      and not tag.startswith('short') and not tag.startswith('pubic') ]
        if len(eyes_tags) == 1 and len(hair_tags) == 1:
            idx.append(infos[0])
            sel_tags = hair_tags + eyes_tags
            tags.append(sel_tags)
    
    idx = np.array(idx, dtype=np.int64)
    tags = np.array(tags)
    return idx, tags

def load_extra_tags(filename):
    tag_file = open(filename, 'r')

    tags = []
    for line in tag_file:
        words = line.strip().split(',')[1].split()
        tags.append([' '.join(words[:2]), ' '.join(words[2:])])

    tags = np.array(tags)
    return tags

def gen_fake_conds(real_conds, num_hair, num_eyes):
    batch_size = real_conds.size(0)
    hair_idx = np.random.randint(num_hair, size=(batch_size,))
    eyes_idx = np.random.randint(num_eyes, size=(batch_size,)) + num_hair

    fake_conds = np.zeros((batch_size, num_hair + num_eyes), dtype=np.float32)
    for i, (h_i, e_i) in enumerate(zip(hair_idx, eyes_idx)):
        if real_conds[i][h_i] == 1. and real_conds[i][e_i] == 1.: h_i = (h_i + 1) % num_hair
        fake_conds[(i, i), (h_i, e_i)] = 1.
            
    fake_conds = torch.FloatTensor(fake_conds)
    return fake_conds

if __name__ == '__main__':
    '''
    jpgfile = Image.open('../data/AnimeDataset/faces/0.jpg')
    jpgfile.thumbnail((64, 64), Image.ANTIALIAS)
    jpgfile = np.array(jpgfile)
    print(jpgfile.shape)
    '''
    '''
    jpgfile = Image.open('../data/extra_data/images/0.jpg')
    jpgfile = np.array(jpgfile)
    print(jpgfile.shape)
    '''
    '''
    import time
    tStart = time.time()
    imgs1 = load_imgs('../data/AnimeDataset/faces/')
    imgs2 = load_imgs('../data/extra_data/images/')
    imgs = np.concatenate((imgs1, imgs2), axis=0)
    tEnd = time.time()
    print(tEnd - tStart)
    print(imgs.shape)
    '''
    '''
    idx, tags1 = load_anime_tags('../../data/AnimeDataset/tags_clean.csv')
    tags2 = load_extra_tags('../../data/extra_data/tags.csv')
    tags = np.concatenate((tags1, tags2))
    hair_feat = np.unique(tags[:, 0])
    eyes_feat = np.unique(tags[:, 1])
    print('hair feat:', len(hair_feat))
    print(hair_feat)
    print('eyes feat:', len(eyes_feat))
    print(eyes_feat)
    '''
    
    img_dir1 = '../../data/AnimeDataset/faces/'
    tag_file1 = '../../data/AnimeDataset/tags_clean.csv'

    img_dir2 = '../../data/extra_data/images/'
    tag_file2 = '../../data/extra_data/tags.csv'
    
    imgs1 = load_imgs(img_dir1)
    idx, tags1 = load_anime_tags(tag_file1)
    
    imgs2 = load_imgs(img_dir2)
    tags2 = load_extra_tags(tag_file2)
    
    imgs = np.concatenate((imgs1[idx], imgs2), axis=0)
    tags = np.concatenate((tags1, tags2), axis=0)
    del imgs1, imgs2, tags1, tags2, idx
    '''
    imgs = load_imgs(img_dir2)
    tags = load_extra_tags(tag_file2)
    '''
    hair_dict = {}
    eyes_dict = {}
    
    for hair_feat in np.unique(tags[:, 0]): hair_dict[hair_feat] = len(hair_dict)
    for eyes_feat in np.unique(tags[:, 1]): eyes_dict[eyes_feat] = len(eyes_dict)
    '''
    tags = np.array([ [ hair_dict[tag[0]], eyes_dict[tag[1]] + len(hair_dict)] for tag in tags ], dtype=np.int64)
    tags = torch.LongTensor(tags)
    '''

    print(len(hair_dict), len(eyes_dict))
    print(hair_dict)
    print(eyes_dict)
    
    myhair_dict = {'aqua hair': 0, 'black hair': 1, 'blonde hair': 2, 'blue hair': 3, 
                   'brown hair': 4, 'gray hair': 5, 'green hair': 6, 'orange hair': 7, 
                   'pink hair': 8, 'purple hair': 9, 'red hair': 10, 'white hair': 11}
    myeyes_dict = {'aqua eyes': 0, 'black eyes': 1, 'blue eyes': 2, 'brown eyes': 3,
                   'green eyes': 4, 'orange eyes': 5, 'pink eyes': 6, 'purple eyes': 7,
                   'red eyes': 8, 'yellow eyes': 9}
    for i, j in zip(hair_dict.keys(), myhair_dict.keys()): 
        if i != j: print(i, j, 'mismatch')
        else: print(i, j)

    for i, j in zip(eyes_dict.keys(), myeyes_dict.keys()): 
        if i != j: print(i, j, 'mismatch')
        else: print(i, j)
