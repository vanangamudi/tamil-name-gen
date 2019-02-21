import os
import re
import sys
import glob
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20 >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np

from functools import partial
from collections import namedtuple, defaultdict, Counter


from anikattu.tokenizer import word_tokenize
from anikattu.tokenstring import TokenString
from anikattu.datafeed import DataFeed, MultiplexedDataFeed
from anikattu.dataset import NLPDataset as Dataset, NLPDatasetList as DatasetList
from anikattu.utilz import tqdm, ListTable
from anikattu.vocab import Vocab
from anikattu.utilz import Var, LongVar, init_hidden, pad_seq
from nltk.tokenize import WordPunctTokenizer
word_punct_tokenizer = WordPunctTokenizer()
word_tokenize = word_punct_tokenizer.tokenize


import tamil

VOCAB =  ['PAD', 'UNK', 'GO', 'EOS']
PAD = VOCAB.index('PAD')

"""
    Local Utilities, Helper Functions

"""
Sample   =  namedtuple('Sample', ['id', 'gender', 'sequence'])

def unicodeToAscii(s):
    import unicodedata
    return ''.join(
        c for c in unicodedata.normalize('NFKC', s)
        if unicodedata.category(c) != 'Mn'
    )

PUNCT_SYMBOLS = '/,<>:;\'"[]{}\|!@#$%^&*()_+-=~` '

def remove_punct_symbols(sentence):
    for i in PUNCT_SYMBOLS:
        sentence = sentence.replace(i, ' ')

    return sentence

def count_UNKS(sentence, vocab):
    return sum(
        [1 for i in sentence if vocab[i] == vocab['UNK']]
    )

def vocab_filter(sentence, vocab):
    return [i if vocab[i] != vocab['UNK'] else 'UNK' for i in sentence ]


class NameDataset(Dataset):
    def __init__(self, name, dataset, input_vocab, gender_vocab, pretrain_samples):
        super().__init__(name, dataset, input_vocab, input_vocab)
        self.gender_vocab = gender_vocab
        self.pretrainset = pretrain_samples

def load_data(config,
              dirname='../raw_dataset/tamil-names',
              max_sample_size=None):


    samples = []
    skipped = 0

    input_vocab = Counter()
    gender_vocab = Counter()

    #########################################################
    # Read names
    #########################################################
    def read_data(filename='names.csv'):
        data = open(filename).readlines()
        samples = []
        for datum in data:
            name = datum.split(',')[1]
            name = ''.join(name.split())
            samples.append(remove_punct_symbols(name))
            
        return samples
    
    def read_dirs(dirs=['boy', 'girl']):
        samples = []
        for d in dirs:
            for filename in os.listdir('{}/{}'.format(dirname, d)):
                s = read_data('{}/{}/{}'.format(dirname, d, filename))
                s = [(d, n) for n in s]
                samples.extend(s)
                
        return list(set(samples))
    

    raw_samples = read_dirs()
    log.info('read {} names'.format(len(raw_samples)))
    
    #########################################################
    # Read tamil words
    #########################################################
    def read_words(filename=config.HPCONFIG.lm_dataset_path):
        samples = []
        for line in tqdm(open(filename).readlines()[:config.HPCONFIG.lm_samples_count],
                         'reading lm file for words'):
            s = line.split()
            s = [('neutral', n) for n in s]
            samples.extend(s)
                
        return list(set(samples))
    

    pretrain_samples = read_words()
    

    #########################################################
    # build vocab
    #########################################################
    all_samples = raw_samples + pretrain_samples
    log.info('building input_vocabulary...')

    for gender, name in tqdm(all_samples, desc='building vocab'):
        name = remove_punct_symbols(name)
        name = tamil.utf8.get_letters(name.strip())
        if len(name):
            input_vocab.update(name)
            gender_vocab.update([gender])


    vocab = Vocab(input_vocab,
                  special_tokens = VOCAB,
                  freq_threshold=50)

    print(gender_vocab)
    gender_vocab = Vocab(gender_vocab,
                         special_tokens = [])
    
    if config.CONFIG.write_vocab_to_file:
        vocab.write_to_file(config.ROOT_DIR + '/input_vocab.csv')
        gender_vocab.write_to_file(config.ROOT_DIR + '/gender_vocab.csv')

    def build_samples(raw_samples):
        samples = []
        for i, (gender, name) in enumerate(
                tqdm(raw_samples, desc='processing names')):
            try:

                #name = remove_punct_symbols(name)
                name = tamil.utf8.get_letters(name.strip())

                if len(name) < 2:
                    continue


                log.debug('===')
                log.debug(pformat(name))

                samples.append(
                    Sample('{}.{}'.format(gender, i),
                           gender,
                           name
                    )
                )

                if  max_sample_size and len(samples) > max_sample_size:
                    break

            except:
                skipped += 1
                log.exception('{}'.format(name))


        return samples
    

    pretrain_samples = build_samples(pretrain_samples)
    samples = build_samples(raw_samples)
    print('skipped {} samples'.format(skipped))

    pivot = int(len(samples) * config.CONFIG.split_ratio)
    train_samples, test_samples = samples[:pivot], samples[pivot:]
    #train_samples, test_samples = samples, []

    #train_samples = sorted(train_samples, key=lambda x: len(x.sequence), reverse=True)
    
    return NameDataset('names',
                       (train_samples, test_samples),
                       pretrain_samples = pretrain_samples,
                       input_vocab = vocab,
                       gender_vocab = gender_vocab)

# ## Loss and accuracy function
def loss(ti, output, batch, loss_function, *args, **kwargs):
    indices, (gender, sequence), _ = batch
    output, state = output
    mask = sequence[:, ti+1] > 0
    output = output * mask.unsqueeze(1).expand_as(output).float()
    return loss_function(output, sequence[:, ti+1]) 

def accuracy(ti, output, batch, *args, **kwargs):
    indices, (gender, sequence,), _ = batch
    output, state = output
    return (output.max(dim=1)[1] == sequence[:, ti+1]).sum().float()/float(answer.size(0))/float(answer.size(1))

def batchop(datapoints, VOCAB, GENDER, config, *args, **kwargs):
    indices = [d.id for d in datapoints]
    sequence = []
    gender = []
    for d in datapoints:
        sequence.append([VOCAB['GO']]
                        + [VOCAB[w] for w in d.sequence]
                        + [VOCAB['EOS']]
        )
        gender.append(GENDER[d.gender])

    sequence    = LongVar(config, pad_seq(sequence))
    gender = LongVar(config, gender)
    batch = indices, (gender, sequence), ()
    return batch
    
def portion(dataset, percent):
    return dataset[ : int(len(dataset) * percent) ]

