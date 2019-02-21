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
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(name)s.%(funcName)s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from functools import partial


import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable

from anikattu.utilz import Var, LongVar, init_hidden, Averager, FLAGS, tqdm
from anikattu.debug import memory_consumed

class Base(nn.Module):
    def __init__(self, config, name):
        super(Base, self).__init__()
        self._name = name
        self.log = logging.getLogger(self._name)
        size_log_name = '{}.{}'.format(self._name, 'size')
        self.log.info('constructing logger: {}'.format(size_log_name))
        self.size_log = logging.getLogger(size_log_name)
        self.size_log.info('size_log')
        self.log.setLevel(config.CONFIG.LOG.MODEL.level)
        self.size_log.setLevel(config.CONFIG.LOG.MODEL.level)
        self.print_instance = 0
        
    def __(self, tensor, name='', print_instance=False):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.__(tensor[i], '{}[{}]'.format(name, i))
        else:
            self.size_log.debug('{} -> {}'.format(name, tensor.size()))
            if self.print_instance or print_instance:
                self.size_log.debug(tensor)

            
        return tensor

    def name(self, n=''):
        if n:
            return '{}.{}'.format(self._name, n)
        else:
            return self._name

        
    def loss_trend(self, metric = None, total_count=10):
        if not metric:
            metric = self.best_model_criteria
            
        if len(metric) > 4:
            losses = metric[-4:]
            count = 0
            for l, r in zip(losses, losses[1:]):
                if l < r:
                    count += 1
                    
            if count > total_count:
                return FLAGS.STOP_TRAINING

        return FLAGS.CONTINUE_TRAINING


    def restore_checkpoint(self):
        try:
            self.snapshot_path = '{}/weights/{}.{}'.format(self.config.ROOT_DIR, self.name(), 'pth')
            self.load_state_dict(torch.load(self.snapshot_path))
            log.info('loaded the old image for the model from :{}'.format(self.snapshot_path))
        except:
            log.exception('failed to load the model  from :{}'.format(self.snapshot_path))

            
    def save_best_model(self):
        with open('{}/{}_best_model_accuracy.txt'.format(self.config.ROOT_DIR, self.name()), 'w') as f:
            f.write(str(self.best_model[0]))

        if self.save_model_weights:
            self.log.info('saving the last best model with accuracy {}...'.format(self.best_model[0]))

            torch.save(self.best_model[1],
                       '{}/weights/{:0.4f}.{}'.format(self.config.ROOT_DIR, self.best_model[0], 'pth'))
            
            torch.save(self.best_model[1],
                       '{}/weights/{}.{}'.format(self.config.ROOT_DIR, self.name(), 'pth'))


    def __build_stats__(self):
        ########################################################################################
        #  Saving model weights
        ########################################################################################
        
        # necessary metrics
        self.mfile_prefix = '{}/results/metrics/{}'.format(self.config.ROOT_DIR, self.name())
        self.train_loss  = Averager(self.config,
                                       filename = '{}.{}'.format(self.mfile_prefix,   'train_loss'))
        
        self.test_loss  = Averager(self.config,
                                        filename = '{}.{}'.format(self.mfile_prefix,   'test_loss'))
        self.accuracy   = Averager(self.config,
                                        filename = '{}.{}'.format(self.mfile_prefix,  'accuracy'))
        
        self.metrics = [self.train_loss, self.test_loss, self.accuracy]
        # optional metrics
        if getattr(self, 'f1score_function'):
            self.tp = Averager(self.config, filename = '{}.{}'.format(self.mfile_prefix,   'tp'))
            self.fp = Averager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'fp'))
            self.fn = Averager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'fn'))
            self.tn = Averager(self.config, filename = '{}.{}'.format(self.mfile_prefix,  'tn'))
            
            self.precision = Averager(self.config,
                                           filename = '{}.{}'.format(self.mfile_prefix,  'precision'))
            self.recall    = Averager(self.config,
                                           filename = '{}.{}'.format(self.mfile_prefix,  'recall'))
            self.f1score   = Averager(self.config,
                                           filename = '{}.{}'.format(self.mfile_prefix,  'f1score'))
          
            self.metrics += [self.tp, self.fp, self.fn, self.tn,
                             self.precision, self.recall, self.f1score]

            
class LM(Base):
    def __init__(self,
                 # config and name
                 config, name,

                 # model parameters
                 vocab_size,
                 gender_vocab_size,

                
                 # feeds
                 dataset,
                 pretrain_feed,
                 train_feed,
                 test_feed,

                 # loss function
                 loss_function,
                 accuracy_function=None,

                 f1score_function=None,
                 save_model_weights=True,
                 epochs = 1000,
                 checkpoint = 1,
                 early_stopping = True,

                 # optimizer
                 optimizer = None,

                 
    ):
        super().__init__(config, name)
        self.config = config
        self.embed_size = config.HPCONFIG.embed_size
        self.hidden_size = config.HPCONFIG.hidden_size
        self.vocab_size = vocab_size
        self.gender_vocab_size = gender_vocab_size
        self.loss_function = loss_function
        
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.gender_embed = nn.Embedding(self.gender_vocab_size, self.embed_size)

        self.position = nn.Linear(1, self.embed_size)
        self.position_range = LongVar(self.config, np.arange(0, 100)).unsqueeze(1).float()
        
        self.blend = nn.Linear(3 * self.embed_size, self. embed_size)
        self.lm  = nn.GRUCell(self.embed.embedding_dim, self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.answer = nn.Linear(self.hidden_size, self.vocab_size)

        self.loss_function = loss_function if loss_function else nn.NLLLoss()
        self.accuracy_function = accuracy_function if accuracy_function else lambda *x, **xx: 1 / loss_function(*x, **xx)

        self.optimizer = optimizer if optimizer else optim.SGD(self.parameters(),lr=0.00001, momentum=0.001)
        self.optimizer = optimizer if optimizer else optim.Adam(self.parameters())
        
        self.f1score_function = f1score_function
        
        self.epochs = epochs
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping

        self.dataset = dataset
        self.pretrain_feed = pretrain_feed
        self.train_feed = train_feed
        self.test_feed = test_feed
        

        ########################################################################################
        #  Saving model weights
        ########################################################################################
        self.save_model_weights = save_model_weights
        self.best_model = (1e5, self.cpu().state_dict())
        try:
            f = '{}/{}_best_model_accuracy.txt'.format(self.config.ROOT_DIR, self.name())
            if os.path.isfile(f):
                self.best_model = (float(open(f).read().strip()), self.cpu().state_dict())
                self.log.info('loaded last best accuracy: {}'.format(self.best_model[0]))
        except:
            log.exception('no last best model')

        self.__build_stats__()
                        
        self.best_model_criteria = self.test_loss
        if config.CONFIG.cuda:
             self.cuda()


    def initial_hidden(self, batch_size):
        ret = Variable(torch.zeros( batch_size, self.lm.hidden_size))
        ret = ret.cuda() if self.config.CONFIG.cuda else ret
        return ret
    
    def forward(self, gender, position, input_, state):
        input_emb  = self.__( self.embed(input_),  'input_emb')
        gender_emb  = self.__( self.gender_embed(gender),  'gender_emb')

        position = self.__(self.position(self.position_range[position]), 'position')
        position = self.__(position.expand_as(gender_emb), 'position')
        
        emb = self.__(self.blend(torch.cat([input_emb, gender_emb, position], dim=-1)), 'blended emb')
        state = self.dropout(self.lm(emb, state))
        return F.log_softmax(self.answer(state), dim=-1), state
        
        
    def do_train(self):
        self.teacher_forcing_ratio = 1
        for epoch in range(self.epochs):

            self.log.critical('memory consumed : {}'.format(memory_consumed()))            
            self.epoch = epoch
            if epoch % max(1, (self.checkpoint - 1)) == 0:
                length = random.randint(5, 10)
                beam_width = random.randint(5, 50)
                self.do_predict(length=length, beam_width=beam_width)
                if self.do_validate() == FLAGS.STOP_TRAINING:
                    self.log.info('loss trend suggests to stop training')
                    return
                           
            self.train()
            teacher_force_count = [0, 0]
            def train_on_feed(feed):

                losses = []
                feed.reset_offset()
                for j in tqdm(range(feed.num_batch), desc='Trainer.{}'.format(self.name())):
                    self.optimizer.zero_grad()
                    input_ = feed.next_batch()
                    idxs, (gender, sequence), targets = input_
                    sequence = sequence.transpose(0,1)
                    _, batch_size = sequence.size()

                    state = self.initial_hidden(batch_size)
                    loss = 0
                    output = sequence[0]
                    for ti in range(1, sequence.size(0) - 1):
                        output = self.forward(gender, ti, output, state)
                        loss += self.loss_function(ti, output, input_)
                        output, state = output

                        if random.random() > self.teacher_forcing_ratio:
                            output = output.max(1)[1]
                            teacher_force_count[0] += 1
                        else:
                            output = sequence[ti+1]
                            teacher_force_count[1] += 1

                    losses.append(loss)
                    loss.backward()
                    self.optimizer.step()
                    
                return torch.stack(losses).mean()

            for i in range(config.HPCONFIG.pretrain_count):
                loss = train_on_feed(self.pretrain_feed)
                
            for i in range(config.HPCONFIG.train_count):
                loss = train_on_feed(self.train_feed)
                self.teacher_forcing_ratio -= 0.1/self.epochs
                
            self.train_loss.append(loss.data.item())                
            self.log.info('teacher_force_count: {}'.format(teacher_force_count))

            self.log.info('-- {} -- loss: {}\n'.format(epoch, self.train_loss))
            
            for m in self.metrics:
                m.write_to_file()

        return True

    def do_validate(self):
        self.eval()
        if self.test_feed.num_batch > 0:
            for j in tqdm(range(self.test_feed.num_batch), desc='Tester.{}'.format(self.name())):
                input_ = self.test_feed.next_batch()
                idxs, (gender, sequence), targets = input_
                sequence = sequence.transpose(0,1)
                _, batch_size = sequence.size()

                state = self.initial_hidden(batch_size)
                loss, accuracy = Var(self.config, [0]), Var(self.config, [0])
                output = sequence[0]
                outputs = []
                ti = 0
                for ti in range(1, sequence.size(0) - 1):
                    output = self.forward(gender, ti, output, state)
                    loss += self.loss_function(ti, output, input_)
                    accuracy += self.accuracy_function(ti, output, input_)
                    output, state = output
                    output = output.max(1)[1]
                    outputs.append(output)

                self.test_loss.append(loss.item())
                if ti == 0: ti = 1
                self.accuracy.append(accuracy.item()/ti)
                #print('====', self.test_loss, self.accuracy)

            self.log.info('= {} =loss:{}'.format(self.epoch, self.test_loss))
            self.log.info('- {} -accuracy:{}'.format(self.epoch, self.accuracy))

            
        if len(self.best_model_criteria) > 1 and self.best_model[0] > self.best_model_criteria[-1]:
            self.log.info('beat best ..')
            self.best_model = (self.best_model_criteria[-1],
                               self.cpu().state_dict())                             

            self.save_best_model()
            
            if self.config.CONFIG.cuda:
                self.cuda()

        
        for m in self.metrics:
            m.write_to_file()
            
        if self.early_stopping:
            return self.loss_trend()
    
    def do_predict(self, input_=None, length=10, beam_width=50):
        self.eval()
        if not input_:
            input_ = self.train_feed.nth_batch(
                random.randint(0, self.train_feed.size - 10),
                1
            )
            
        idxs, (gender, sequence), targets = input_
        sequence = sequence.transpose(0,1)
        _, batch_size = sequence.size()
        
        state = self.initial_hidden(batch_size)
        loss = 0
        output = sequence[1]
        outputs = []
        for ti in range(length - 1):
            outputs.append(output)
            output = self.forward(gender, ti, output, state)
            output, state = output
            output = output.topk(beam_width)[1]
            index = random.randint(0, beam_width-1)
            output = output[:, index]

        outputs = torch.stack(outputs).transpose(0,1)
        for i in range(outputs.size(0)):

            s = [self.dataset.input_vocab[outputs[i][j]] for j in range(outputs.size(1))]
            print(self.dataset.gender_vocab[gender.item()], ''.join(s), length, beam_width)
            
        return True
