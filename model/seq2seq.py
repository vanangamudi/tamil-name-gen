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

from .lm import Base

class Seq2SeqModel(Base):
    def __init__(self, config, name,
                 input_vocab_size,
                 gender_vocab_size,
                 output_vocab_size,
    
                 # feeds
                 dataset,
                 pretrain_feed,
                 train_feed,
                 test_feed,

                 # loss function
                 loss_function,
                 accuracy_function,

                 f1score_function=None,
                 save_model_weights=True,
                 epochs = 1000,
                 checkpoint = 1,
                 early_stopping = True,

                 # optimizer
                 optimizer = None,):
        
        
        super().__init__(config, name)
        self.vocab_size = input_vocab_size
        self.gender_vocab_size = input_vocab_size
        self.hidden_dim = config.HPCONFIG.hidden_dim
        self.embed_dim = config.HPCONFIG.embed_dim

        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.gender_embed = nn.Embedding(self.gender_vocab_size, self.embed_dim)
                        
        self.encode = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        self.decoder = nn.LSTMCell(self.embed_dim + self.hidden_dim, self.hidden_dim)

        self.classify = nn.Linear(self.hidden_dim, self.vocab_size)

        self.loss_function = loss_function if loss_function else nn.NLLLoss()
        self.accuracy_function = accuracy_function
        self.optimizer = optimizer if optimizer else optim.SGD(self.parameters(),
                                                               lr=self.config.HPCONFIG.LR,
                                                               momentum=self.config.HPCONFIG.MOMENTUM)
        self.optimizer = optimizer if optimizer else optim.Adam(self.parameters())
        
        self.f1score_function = f1score_function
        
        self.epochs = epochs
        self.checkpoint = checkpoint
        self.early_stopping = early_stopping

        self.dataset = dataset
        self.pretrain_feed = pretrain_feed
        self.train_feed = train_feed
        self.test_feed = test_feed
        
        self.save_model_weights = save_model_weights

        self.__build_stats__()
                        
        self.best_model_criteria = self.test_loss
        self.best_model = (1e+10,
                           self.cpu().state_dict())  

        if config.CONFIG.cuda:
            self.cuda()
        
    def restore_and_save(self):
        self.restore_checkpoint()
        self.save_best_model()
        
    def init_hidden(self, batch_size):
        hidden  = Variable(torch.zeros(batch_size, self.hidden_dim))
        cell  = Variable(torch.zeros(batch_size, self.hidden_dim))
        if config.CONFIG.cuda:
            hidden  = hidden.cuda()
            cell    = cell.cuda()
            
        return hidden, cell

    def encode_sequence(self, seq):
        seq_size, batch_size = seq.size()
        hidden_states = []
        seq_emb = self.__(self.embed(seq), 'seq_emb')
        hidden, cell_state = self.init_hidden(batch_size)
        for index in range(seq_size):
            hidden, cell_state = self.encode(seq_emb[index], (hidden, cell_state)) 
            hidden_states.append(hidden)
            
        return torch.stack(hidden_states), (hidden, cell_state)
    
    def decode(self, encoder_states, prev_output, state, gender_embedding):
        prev_output_emb = self.__( self.embed(prev_output), 'prev_output_emb' )

        hidden, cell_state = self.__(state, 'state')

        input_ = torch.cat([prev_output_emb, gender_embedding], dim=-1)
        hidden, cell_state = self.decoder(input_, state) 
        logits = self.classify(hidden)        
        return F.log_softmax(logits, dim=-1), (hidden, cell_state)
   
    def do_train(self):

        for epoch in range(self.epochs):
            self.log.critical('memory consumed : {}'.format(memory_consumed()))            
            self.epoch = epoch
            if epoch % max(1, (self.checkpoint - 1)) == 0:
                self.do_predict()
                if self.do_validate() == FLAGS.STOP_TRAINING:
                    self.log.info('loss trend suggests to stop training')
                    return
                
            self.train()
            def train_on_feed(feed):
                losses = []
                feed.reset_offset()
                                
                for j in tqdm(range(feed.num_batch), desc='Trainer.{}'.format(self.name())):
                    self.optimizer.zero_grad()
                    input_ = feed.next_batch()
                    idxs, (gender, seq), target = input_

                    seq_size, batch_size = seq.size()
                    pad_mask = (seq > 0).float()

                    hidden_states, (hidden, cell_state) = self.__(self.encode_sequence(seq),
                                                                  'encoded_outpus')

                    loss = 0
                    outputs = []
                    target_size, batch_size = target.size()
                    #TODO: target[0] should not be used. will throw error when used without GO token from batchip
                    output = self.__(target[0], 'hidden')
                    state = self.__((hidden, cell_state), 'init_hidden')
                    gender_embedding = self.gender_embed(gender)
                    for index in range(target_size - 1):
                        output, state = self.__(self.decode(hidden_states,
                                                            output, state,
                                                            gender_embedding),
                                                'output, state')
                        loss   += self.loss_function(output, target[index+1])
                        output = self.__(output.max(1)[1], 'output')
                        outputs.append(output)

                    losses.append(loss)
                    loss.backward()
                    self.optimizer.step()

                return torch.stack(losses).mean()
            
            for i in range(config.HPCONFIG.pretrain_count):
                loss = train_on_feed(self.pretrain_feed)
                
            for i in range(config.HPCONFIG.train_count):
                loss = train_on_feed(self.train_feed)
                self.log.info('-- {} -- loss: {}\n'.format(epoch, loss.item()))
                
            self.train_loss.append(loss.data.item())                
            self.log.info('-- {} -- best loss: {}\n'.format(epoch, self.best_model[0]))
            
            for m in self.metrics:
                m.write_to_file()

        return True

    def do_validate(self):
        self.eval()
        
        if self.test_feed.num_batch > 0:
            
            losses, accuracies = [], []
            print('testset offset :', self.test_feed.offset)
            for j in tqdm(range(self.test_feed.num_batch), desc='Tester.{}'.format(self.name())):
                input_ = self.test_feed.next_batch()
                idxs, (gender, seq), target = input_

                seq_size, batch_size = seq.size()
                pad_mask = (seq > 0).float()

                hidden_states, (hidden, cell_state) = self.__(self.encode_sequence(seq),
                                                              'encoded_outpus')
                
                loss = 0
                accuracy = 0
                outputs = []
                target_size, batch_size = target.size()
                #TODO: target[0] should not be used. will throw error when used without GO token from batchip
                output = self.__(target[0], 'hidden')
                state = self.__((hidden, cell_state), 'init_hidden')
                gender_embedding = self.gender_embed(gender)                            
                        
                for index in range(target_size - 1):
                    output, state = self.__(self.decode(hidden_states,
                                                        output, state,
                                                        gender_embedding),
                                            'output, state')
                    loss   += self.loss_function(output, target[index+1])
                    accuracy   += self.accuracy_function(output, target[index+1])
                    output = self.__(output.max(1)[1], 'output')
                    outputs.append(output)
                    
                losses.append(loss.detach())
                accuracies.append(accuracy.detach())

            epoch_loss = torch.stack(losses).mean() / target_size
            epoch_accuracy = torch.stack(accuracies).mean() / target_size
            self.test_loss.append(epoch_loss.data.item())
            self.accuracy.append(epoch_accuracy.data.item())

            self.log.info('= {} =loss:{}'.format(self.epoch, epoch_loss))
            self.log.info('= {} =accuracy:{}'.format(self.epoch, epoch_accuracy))
            
        if len(self.best_model_criteria) > 1:
            if self.best_model[0] > self.best_model_criteria[-1]:
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
    
    def do_predict(self, input_=None, max_len=100, length=10, beam_width=4, teacher_force=False):
        if not input_:
            input_ = self.train_feed.nth_batch(
                random.randint(0, self.train_feed.size),
                1
            )

        idxs, (gender, seq), target = input_

        #seq = seq[1:]
        #target = target[1:]
        seq_size, batch_size = seq.size()
        pad_mask = (seq > 0).float()
        
        hidden_states, (hidden, cell_state) = self.__(self.encode_sequence(seq),
                                                      'encoded_outpus')

        outputs = []
        target_size, batch_size = seq.size()
        output = self.__(target[0], 'hidden')
        outputs.append(output)

        state = self.__((hidden, cell_state), 'init_hidden')
        gender_embedding = self.gender_embed(gender)
        null_tensor = LongVar(self.config, [self.dataset.input_vocab['_']])
        for i in range(1, target_size):
            output, state = self.__(self.decode(hidden_states,
                                                output, state,
                                                gender_embedding),
                                    
                                    'output, state')
            output = output.topk(beam_width)[1]
            index = random.randint(0, beam_width-1)

            output = output[:, index]
            if teacher_force:
                #teacher force only where non '_' characters are given
                if seq[i].eq(null_tensor.expand_as(seq[i])).sum().float() < 0.5:
                    #print(seq[i].eq(null_tensor.expand_as(seq[i])).sum())
                    output = seq[i]
                    #print(self.dataset.input_vocab[seq[i][0]])
            outputs.append(output)
            
        outputs = torch.stack(outputs).long().t()
        seq = seq.t()
        #print(output.size())
        #print(seq.size())
        #print(target.size())

        print(''.join([self.dataset.input_vocab[i.item()] for i in target[1:-1]]), end='\t')
        print(''.join([self.dataset.input_vocab[i.item()] for i in seq[0][1:-1]]), end='\t')
        print(''.join([self.dataset.input_vocab[i.item()] for i in outputs[0][1:-1]]))
        
        return True

    

class AttnSeq2SeqModel(Seq2SeqModel):
    def __init__(self, config, name,
                 input_vocab_size,
                 gender_vocab_size,
                 output_vocab_size,
    
                 # feeds
                 dataset,
                 pretrain_feed,
                 train_feed,
                 test_feed,

                 # loss function
                 loss_function,
                 accuracy_function,

                 f1score_function=None,
                 save_model_weights=True,
                 epochs = 1000,
                 checkpoint = 1,
                 early_stopping = True,

                 # optimizer
                 optimizer = None,):
        
        
        super().__init__(config, name,
                         input_vocab_size,
                         gender_vocab_size,
                         output_vocab_size,
                         
                         # feeds
                         dataset,
                         pretrain_feed,
                         train_feed,
                         test_feed,
                         
                         # loss function
                         loss_function,
                         accuracy_function,
                         
                         f1score_function,
                         save_model_weights,
                         epochs,
                         checkpoint,
                         early_stopping,
                         
                         # optimizer
                         optimizer,)
        

        self.attn = nn.Parameter(torch.zeros([self.hidden_dim, self.hidden_dim]))
        self.decoder = nn.LSTMCell(self.embed_dim + self.hidden_dim + self.hidden_dim, self.hidden_dim)
                
        self.best_model_criteria = self.train_loss
        self.best_model = (1e+10,
                           self.cpu().state_dict())  

        if config.CONFIG.cuda:
            self.cuda()
        
    def decode(self, encoder_states, prev_output, state, gender_embedding):
        prev_output_emb = self.__( self.embed(prev_output), 'prev_output_emb' )

        hidden, cell_state = self.__(state, 'state')
        hidden_ = self.__(hidden.unsqueeze(-1), 'hidden_')

        batch_size, _ = hidden.size()
        
        attn = self.__(self.attn.expand(batch_size, *self.attn.size()), 'attn')
        attn    = self.__(torch.bmm(attn, hidden_) , 'attn')

        encoder_states = self.__(encoder_states.transpose(0,1).transpose(1,2), 'encoder_states')
        attn = self.__(torch.bmm(attn.transpose(1, 2), encoder_states), 'attn')

        attn = self.__(F.softmax(attn, dim=-1), 'attn')
        attn_state = self.__(attn.expand_as(encoder_states) * encoder_states, 'attn_states')
        
        input_ = torch.cat([prev_output_emb, attn_state.sum(dim=-1), gender_embedding], dim=-1)
        hidden, cell_state = self.decoder(input_, state) 
        logits = self.classify(hidden)        
        return F.log_softmax(logits, dim=-1), (hidden, cell_state)
   
