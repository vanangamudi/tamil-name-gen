import os
import re
import sys
import json
import time
import random
from pprint import pprint, pformat

sys.path.append('..')
import config
from anikattu.logger import CMDFilter
import logging
logging.basicConfig(format=config.FORMAT_STRING)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
import sys


from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from anikattu.utilz import initialize_task, tqdm

from model.lm import LM
from utilz import load_data, batchop, loss, portion, Sample

import importlib


SELF_NAME = os.path.basename(__file__).replace('.py', '')

import sys
import pickle
import argparse
from matplotlib import pyplot as plt
from functools import partial
plt.style.use('ggplot')



from anikattu.tokenizer import word_tokenize
from anikattu.tokenstring import TokenString
from anikattu.datafeed import DataFeed, MultiplexedDataFeed
from anikattu.dataset import NLPDataset as Dataset, NLPDatasetList as DatasetList
from anikattu.utilz import tqdm, ListTable
from anikattu.vocab import Vocab
from anikattu.utilz import Var, LongVar, init_hidden, pad_seq, dump_vocab_tsv
from nltk.tokenize import WordPunctTokenizer
word_punct_tokenizer = WordPunctTokenizer()
word_tokenize = word_punct_tokenizer.tokenize



if __name__ == '__main__':
    start = time.time()

    ########################################################################################
    # Parser arguments
    ########################################################################################
    parser = argparse.ArgumentParser(description='MACNet variant 2')
    parser.add_argument('-p','--hpconfig',
                        help='path to the hyperparameters config file',
                        default='hpconfig.py', dest='hpconfig')
    
    parser.add_argument('-b','--base-hpconfig',
                        help='path to the base hyperparameters config file',
                        default=None, dest='base_hpconfig')
        
    parser.add_argument('-d', '--prefix-dir',
                        help='path to the results',
                        default='run00', dest='prefix_dir')
    
    parser.add_argument('--log-filters',
                        help='log filters',
                        dest='log_filter')

    subparsers = parser.add_subparsers(help='commands')

    donothing_parser = subparsers.add_parser('donothing', help='does nothing')
    donothing_parser.add_argument('--donothing', default='donothing', dest='task')
    
    dump_vocab_parser = subparsers.add_parser('dump-vocab',
                                              help='dumps the vocabulary into two tsv file')
    dump_vocab_parser.add_argument('--dump-vocab', default='dump-vocab', dest='task')
    dump_vocab_parser.add_argument('--mux', action='store_true', default=False, dest='mux')

    
    train_parser = subparsers.add_parser('train', help='starts training')
    train_parser.add_argument('--train', default='train', dest='task')
    train_parser.add_argument('--mux', action='store_true', default=False, dest='mux')
    
    predict_parser = subparsers.add_parser('predict',
                                help='''starts a cli interface for running predictions 
                                in inputs with best model from last training run''')

    predict_parser.add_argument('--predict', default='predict', dest='task')
    predict_parser.add_argument('-u', '--uniqueness', default=50,  dest='beam_width', type=int)
    predict_parser.add_argument('-c', '--count',  default=10, dest='count', type=int)
    predict_parser.add_argument('-l', '--length', default=10, dest='prediction_length', type=int)
    predict_parser.add_argument('-g', '--gender', default='girl', dest='gender')
    predict_parser.add_argument('-s', '--starting-char', default=None, dest='starting_char')
    
    predict_parser.add_argument('--over-test-feed', action='store_true', dest='over_test_feed')
    predict_parser.add_argument('--show-plot', action='store_true', dest='show_plot')
    predict_parser.add_argument('--save-plot', action='store_true',  dest='save_plot')


    args = parser.parse_args()
    print(args)
    if args.log_filter:
        log.addFilter(CMDFilter(args.log_filter))

        
    ########################################################################################
    # anikattu initialization for directory structure and so on
    ########################################################################################
    ROOT_DIR = initialize_task(args.hpconfig, args.prefix_dir, base_hpconfig=args.base_hpconfig)

    sys.path.append('.')
    print(sys.path)
    HPCONFIG = importlib.__import__(args.hpconfig.replace('.py', ''))
    config.HPCONFIG = HPCONFIG.CONFIG
    config.ROOT_DIR = ROOT_DIR
    config.NAME = SELF_NAME
    print('====================================')
    print(ROOT_DIR)
    print('====================================')

    ########################################################################################
    # flush and load dataset or restore pickle file
    ########################################################################################
    if config.CONFIG.flush:
        log.info('flushing...')
        dataset = load_data(config, dirname=config.HPCONFIG.dataset_path)
        pickle.dump(dataset, open('{}__cache.pkl'.format(SELF_NAME), 'wb'))
    else:
        dataset = pickle.load(open('{}__cache.pkl'.format(SELF_NAME), 'rb'))
        
    log.info('dataset size: {}'.format(len(dataset.trainset)))
    log.info('dataset[:10]: {}'.format(pformat(random.choice(dataset.trainset))))

    #log.info('vocab: {}'.format(pformat(dataset.output_vocab.freq_dict)))
    ########################################################################################
    # load model snapshot data 
    ########################################################################################
    _batchop = partial(batchop, VOCAB=dataset.input_vocab, GENDER=dataset.gender_vocab, config=config)
    train_feed     = DataFeed(SELF_NAME,
                              dataset.trainset,
                              batchop    = _batchop,
                              batch_size = config.CONFIG.batch_size)

    pretrain_feed     = DataFeed(SELF_NAME,
                                 dataset.pretrainset,
                                 batchop    = _batchop,
                                 batch_size = config.CONFIG.batch_size)
    
    
    loss_ = partial(loss, loss_function=nn.NLLLoss())
    test_feed      = DataFeed(SELF_NAME, dataset.testset, batchop=_batchop, batch_size=config.CONFIG.batch_size)
    model =  LM(config, 'LM',
                len(dataset.input_vocab),
                gender_vocab_size = len(dataset.gender_vocab),
                loss_function = loss_,
                dataset = dataset,
                pretrain_feed = pretrain_feed,
                train_feed = train_feed,
                test_feed = test_feed,
    )

    model.restore_checkpoint()
    
    if config.CONFIG.cuda:
        model = model.cuda()        
        if config.CONFIG.multi_gpu and torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)


    
    print('**** the model', model)    
    if args.task == 'train':
        model.do_train()
        
    if args.task == 'dump-vocab':
        dump_vocab_tsv(config,
                   dataset.input_vocab,
                   model.embed.weight.data.cpu().numpy(),
                   config.ROOT_DIR + '/vocab.tsv')


    if args.task == 'predict':
        if args.starting_char:
            datapoints = [Sample(0, args.gender, [args.starting_char])]
            input_ = _batchop(datapoints)
        else:
            input_ = None
            
        for i in range(args.count):
            try:
                
                output = model.do_predict(input_,
                                          length=args.prediction_length,
                                          beam_width=args.beam_width)
            except:
                log.exception('#########3')
                pass
                
    end = time.time()
    print('{} ELAPSED: {}'.format(ROOT_DIR, end - start))
