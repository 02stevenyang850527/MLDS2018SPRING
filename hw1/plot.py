import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(prog='plot.py', description='MLDS2018 hw1-1.1 Stimulate Functions')
parser.add_argument('--sr', type=int, default=1000)
parser.add_argument('--rng', type=list, default=[0., 5.])

parser.add_argument('--mode', type=str, default='hw1-1-1')
parser.add_argument('--func', type=str, default='damping')
parser.add_argument('--save_dir', type=str, default='./')
parser.add_argument('--result_dir', type=str, default='./')
args = parser.parse_args()

if args.mode == 'hw1-1-1':
    def target_func(x):
        if args.func == 'damping': return math.exp(-x) * math.cos(2 * math.pi * x)
        elif args.func == 'triangle': return 1. if x % 2 < 1 else 0.

    if args.func == 'damping':
        collection = {'shallow': args.result_dir + 'damping_shallow.csv',
                      'medium' : args.result_dir + 'damping_medium.csv', 
                      'deep'   : args.result_dir + 'damping_deep.csv' }

    elif args.func == 'triangle':
        collection = {'shallow': args.result_dir + 'triangle_shallow.csv', 
                      'medium' : args.result_dir + 'triangle_medium.csv', 
                      'deep'   : args.result_dir + 'triangle_deep.csv' }

    plt.figure()

    for model, path in collection.items():
        pltfile = open(path, 'r')
        data = pd.read_csv(pltfile)
        x, y = data['x'].values, data['f(x)'].values
        plt.plot(x, y, label=model)

    x = np.arange(*args.rng, 1 / args.sr, dtype=np.float32)
    y = np.array([target_func(i) for i in x], dtype=np.float32)
    plt.plot(x, y, label='origin')

    plt.xlabel('x')
    plt.ylabel('f(x)')

    if args.func == 'damping':
        plt.title('Damping Function')
        plt.legend()
        plt.savefig(args.result_dir + 'damping.png')

    elif args.func == 'triangle':
        plt.title('Triangle Wave Function')
        plt.legend()
        plt.savefig(args.result_dir + 'triangle.png')

        

    if args.func == 'damping':
        collection = {'shallow': args.save_dir + 'damping_shallow_loss.csv', 
                      'medium' : args.save_dir + 'damping_medium_loss.csv', 
                      'deep'   : args.save_dir + 'damping_deep_loss.csv' }

    elif args.func == 'triangle':
        collection = {'shallow': args.save_dir +'triangle_shallow_loss.csv', 
                      'medium' : args.save_dir +'triangle_medium_loss.csv', 
                      'deep'   : args.save_dir +'triangle_deep_loss.csv' }

    plt.figure()

    for model, path in collection.items():
        pltfile = open(path, 'r')
        data = pd.read_csv(pltfile)
        x, y = data['epoch'].values, data['loss'].values
        plt.plot(x, y, label=model)

    plt.xlabel('# of Epochs')
    plt.ylabel('Training Loss')

    if args.func == 'damping':
        plt.title('Damping Function')
        plt.legend()
        plt.savefig(args.result_dir +'damping_loss.png')

    elif args.func == 'triangle':
        plt.title('Triangle Wave Function')
        plt.legend()
        plt.savefig(args.result_dir + 'triangle_loss.png')



if args.mode == 'hw1-2-3':
    pltfile = open(args.save_dir + 'minimal_ratio_loss.csv', 'r')
    data = pd.read_csv(pltfile)
    x, y = data['minimal ratio'].values, data['loss'].values
    plt.plot(x, y, '.')

    plt.xlabel('minimal ratio')
    plt.ylabel('loss')
    plt.title('Damping Function')
    plt.savefig(args.result_dir + 'minimal_ratio.png')



if args.mode == 'hw1-3-3-part2':
    pltfile = open(args.save_dir + 'sensitivity_loss.csv', 'r')
    data = pd.read_csv(pltfile)
    x = data['batch'].values
    y1 = data['sensitivity'].values
    y2 = data['testing loss'].values
    y3 = data['training loss'].values

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xscale('log', nonposx='clip')
    ax1.set_yscale('log', nonposy='clip')
    ax1.plot(x, y3, 'b',label='train' )
    ax1.plot(x, y2, '--b', label='test')
    ax1.set_ylabel('cross entropy (log scale)')
    ax1.set_xlabel('batch size (log scale)')
    ax1.yaxis.label.set_color('blue')
    ax1.tick_params(axis='y', colors='blue')
    ax1.legend(loc=2)

    ax2 = ax1.twinx()
    ax2.set_xscale('log', nonposx='clip')
    ax2.plot(x, y1, 'r', label='sensitivity')
    ax2.set_ylabel('sensitivity')
    ax2.set_xlabel('batch size (log scale)')
    ax2.yaxis.label.set_color('red')
    ax2.tick_params(axis='y', colors='red')
    ax2.legend(loc=1)

    plt.savefig(args.result_dir + 'sensitivity_loss.png')



    y2 = data['testing accuracy'].values
    y3 = data['training accuracy'].values

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xscale('log', nonposx='clip')
    ax1.plot(x, y3, 'b',label='train' )
    ax1.plot(x, y2, '--b', label='test')
    ax1.set_ylabel('accuracy')
    ax1.set_xlabel('batch size (log scale)')
    ax1.yaxis.label.set_color('blue')
    ax1.tick_params(axis='y', colors='blue')
    ax1.legend(loc=2)

    ax2 = ax1.twinx()
    ax2.set_xscale('log', nonposx='clip')
    ax2.plot(x, y1, 'r', label='sensitivity')
    ax2.set_ylabel('sensitivity')
    ax2.set_xlabel('batch size (log scale)')
    ax2.yaxis.label.set_color('red')
    ax2.tick_params(axis='y', colors='red')
    ax2.legend(loc=1)

    plt.savefig(args.result_dir + 'sensitivity_acc.png')
