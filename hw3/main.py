import matplotlib
matplotlib.use('Agg')  # if necessary
import os
import time
import argparse
import numpy as np 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from model import weights_init, Generator, Discriminator
from utils import USE_CUDA, stdout, load_imgs, load_anime_tags, load_extra_tags, gen_fake_conds


parser = argparse.ArgumentParser(prog='main.py', description='MLDS2018 hw3.2 Conditional GAN')
parser.add_argument('--test' , action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)

parser.add_argument('--lr'      , type=float, default=2e-4)
parser.add_argument('--beta1'   , type=float, default=0.9)
parser.add_argument('--epoch'   , type=int  , default=500)
parser.add_argument('--batch'   , type=int  , default=-1)
parser.add_argument('--model'   , type=str  , default='cgan')
parser.add_argument('--conti'   , type=int  , default=0)
parser.add_argument('--noise'   , type=int  , default=250)
parser.add_argument('--filter'  , type=int  , default=32)
parser.add_argument('--period'  , type=int  , default=1) 
parser.add_argument('--channel' , type=int  , default=3)

parser.add_argument('--prefix'     , type=str, default='unnamed')
parser.add_argument('--data_dir'   , type=str, default='../../data/')
parser.add_argument('--save_dir'   , type=str, default='./save/')
parser.add_argument('--test_tag'   , type=str, default=None)
parser.add_argument('--noise_file' , type=str, default=None)
parser.add_argument('--result_dir' , type=str, default='../result/')
parser.add_argument('--result_file', type=str, default=None)
args = parser.parse_args()

result_prefix = args.result_dir + args.prefix
save_prefix = args.save_dir + args.prefix

img_dir1 = args.data_dir + 'AnimeDataset/faces/'
tag_file1 = args.data_dir + 'AnimeDataset/tags_clean.csv'

img_dir2 = args.data_dir + 'extra_data/images/'
tag_file2 = args.data_dir + 'extra_data/tags.csv'

val_tag_file = args.data_dir + 'AnimeDataset/testing_tags.txt'


if args.test == False and args.train == False:
    print('Error: can\'t find argument --train or --test.')
    os._exit(0)


hair_dict = {'aqua hair': 0, 'black hair': 1, 'blonde hair': 2, 'blue hair': 3, 
             'brown hair': 4, 'gray hair': 5, 'green hair': 6, 'orange hair': 7, 
             'pink hair': 8, 'purple hair': 9, 'red hair': 10, 'white hair': 11}
eyes_dict = {'aqua eyes': 0, 'black eyes': 1, 'blue eyes': 2, 'brown eyes': 3,
             'green eyes': 4, 'orange eyes': 5, 'pink eyes': 6, 'purple eyes': 7,
             'red eyes': 8, 'yellow eyes': 9}
num_hair = len(hair_dict)
num_eyes = len(eyes_dict)

if args.train:
    if not os.path.exists(result_prefix): os.mkdir(result_prefix)
    if not os.path.exists(save_prefix): os.mkdir(save_prefix)
    logger_path = save_prefix + '.csv'
    
    if args.conti != 0: logger_file = open(logger_path, 'a')
    else: 
        logger_file = open(logger_path, 'w')
        print('epoch,loss_D,loss_G', file=logger_file)
  
    '''
    imgs1 = load_imgs(img_dir1)
    idx, tags1 = load_anime_tags(tag_file1)
    imgs2 = load_imgs(img_dir2)
    tags2 = load_extra_tags(tag_file2)
    
    imgs = np.concatenate((imgs1[idx], imgs2), axis=0)
    tags = np.concatenate((tags1, tags2), axis=0)
    del imgs1, imgs2, tags1, tags2, idx
    
    ''' 
    ### only extra data ###
    imgs = load_imgs(img_dir2)
    tags = load_extra_tags(tag_file2)
    
    '''
    ### for img flip ###
    flip_imgs = np.array([ np.fliplr(img) for img in imgs ])
    imgs = np.concatenate((imgs, flip_imgs), axis=0)
    tags = np.tile(tags, (2, 1))
    '''
    imgs = torch.FloatTensor(imgs).permute(0, 3, 1, 2)
    

    conds = torch.zeros((imgs.size(0), num_hair + num_eyes))
    for i, tag in enumerate(tags): conds[(i, i), (hair_dict[tag[0]], eyes_dict[tag[1]] + num_hair)] = 1.
    
    dataset = TensorDataset(imgs, conds)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=3)
    
    val_tags = load_extra_tags(val_tag_file)
    val_conds = torch.zeros((len(val_tags), num_hair + num_eyes))
    for i, tag in enumerate(val_tags): val_conds[(i, i), (hair_dict[tag[0]], eyes_dict[tag[1]] + num_hair)] = 1.
    val_noise = torch.Tensor(len(val_tags), args.noise).uniform_(-1.2, 1.2)
    
    val_noise = Variable(val_noise).cuda() if USE_CUDA else Variable(val_noise)
    val_conds = Variable(val_conds).cuda() if USE_CUDA else Variable(val_conds)
    
    if args.conti != 0:
        netG = torch.load(save_prefix + '/G_epoch_{}.pkl'.format(args.conti))
        netD = torch.load(save_prefix + '/D_epoch_{}.pkl'.format(args.conti))
    else:
        if args.model.lower() == 'cgan':
            netG = Generator(args.noise, num_hair + num_eyes, args.filter*2, args.channel)
            netD = Discriminator(num_hair + num_eyes, args.channel, args.filter)
            if USE_CUDA: netG, netD = netG.cuda(), netD.cuda()
            netG.apply(weights_init)
            netD.apply(weights_init)
            

    optG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    criterion = nn.BCELoss()
    
    total_steps = max(int(np.ceil(imgs.size(0)/args.batch)), 1)

    for n_epoch in range(args.conti, args.conti + args.epoch):
        netG.train()
        netD.train()
        
        step_loss_D = []
        step_loss_G = []
        cost = 0
        for n_step, (real_x, real_conds) in enumerate(loader):
            tStart = time.time()
            optD.zero_grad()
            
            real_y = torch.ones((real_x.size(0),))
            fake_y = torch.zeros((real_x.size(0),))
            
            tr_noise = torch.Tensor(real_x.size(0), args.noise).uniform_(-1.2, 1.2)
            fake_conds = gen_fake_conds(real_conds, num_hair, num_eyes)

            real_x = Variable(real_x).cuda() if USE_CUDA else Variable(real_x)
            real_y = Variable(real_y).cuda() if USE_CUDA else Variable(real_y)
            fake_y = Variable(fake_y).cuda() if USE_CUDA else Variable(fake_y)
            tr_noise = Variable(tr_noise).cuda() if USE_CUDA else Variable(tr_noise)
            real_conds = Variable(real_conds).cuda() if USE_CUDA else Variable(real_conds)
            fake_conds = Variable(fake_conds).cuda() if USE_CUDA else Variable(fake_conds)
            

            outputs = netD(real_x, real_conds)
            real_loss = criterion(outputs, real_y)
            
            outputs = netD(real_x, fake_conds)
            fake_loss = criterion(outputs, fake_y)

            gen_x = netG(tr_noise, real_conds)
            outputs = netD(gen_x.detach(), real_conds)
            gen_loss = criterion(outputs, fake_y)
            
            loss = real_loss + (fake_loss + gen_loss) / 2 
            step_loss_D.append(loss.data[0])
            loss.backward()
            optD.step()

            
            optG.zero_grad()
            fake_y.fill_(1.)
            outputs = netD(gen_x, real_conds)
            #loss = criterion(outputs, real_y)
            loss = criterion(outputs, fake_y)
            step_loss_G.append(loss.data[0])
            loss.backward()
            optG.step()

            tEnd = time.time()

            stdout(n_epoch, n_step, total_steps, tEnd-tStart, step_loss_D[-1], step_loss_G[-1])
            cost += tEnd - tStart
        
        epoch_loss_G = np.mean(step_loss_G)
        epoch_loss_D = np.mean(step_loss_D)
        stdout(n_epoch, total_steps, total_steps, cost, epoch_loss_D, epoch_loss_G)
        print(n_epoch+1, epoch_loss_D, epoch_loss_G, file=logger_file)
        
        netG.eval()
        netD.eval()
        
        val_x = netG(val_noise, val_conds)
        
        if (n_epoch + 1) % 10 == 0:
            params_G_path = save_prefix + '/G_epoch_{}.pkl'.format(n_epoch + 1)
            params_D_path = save_prefix + '/D_epoch_{}.pkl'.format(n_epoch + 1)
            torch.save(netG, params_G_path)
            torch.save(netD, params_D_path)
        
        result_file = result_prefix + '/epoch_{}.png'.format(n_epoch + 1)
        vutils.save_image(val_x.data, result_file, normalize=True)
             
    logger_file.close()

if args.test:
    netG = torch.load(save_prefix + '_G.pkl')
    if USE_CUDA: netG = netG.cuda()
    
    if args.test_tag != None:
        tt_tags = load_extra_tags(args.test_tag)
    else:
        val_tags = load_extra_tags(val_tag_file)

    if args.noise_file != None:
        noise_file = open(args.noise_file, 'r')
        tt_noise = np.array([ row.strip().split() for row in noise_file ], dtype=np.float32)
        tt_noise = torch.FloatTensor(tt_noise)
        if len(tt_tags) != tt_noise.size(0):
            tt_noise = torch.Tensor(len(tt_tags), args.noise).uniform_(-1.2, 1.2)
    else:
        tt_noise = torch.Tensor(len(tt_tags), args.noise).uniform_(-1.2, 1.2)
    
    tt_conds = torch.zeros((len(tt_tags), num_hair + num_eyes))
    for i, tag in enumerate(tt_tags): tt_conds[(i, i), (hair_dict[tag[0]], eyes_dict[tag[1]] + num_hair)] = 1.
    
    tt_noise = Variable(tt_noise).cuda() if USE_CUDA else Variable(tt_noise)
    tt_conds = Variable(tt_conds).cuda() if USE_CUDA else Variable(tt_conds)
    
    tt_x = netG(tt_noise, tt_conds)
    tt_x = tt_x.permute(0, 2, 3, 1)
    tt_x = tt_x.cpu().data.numpy() if USE_CUDA else tt_x.data.numpy()
    tt_x = np.clip(tt_x * 255., 0., 255.).astype(np.uint8)

    r, c = 5, 5
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(tt_x[cnt, :, :, :])
            axs[i, j].axis('off')
            cnt += 1
     
    fig.savefig(args.result_file)
    plt.close()
