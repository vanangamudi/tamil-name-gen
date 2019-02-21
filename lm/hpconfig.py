import logging

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):
    dataset = 'names_'
    dataset_path = '../dataset/'
    lm_dataset_path = '../dataset/lm_lengthsorted.txt'
    lm_samples_count = 1000000
    trainset_size = 1.0
    max_story_len = 0
    hidden_size = 50
    embed_size = 50
    num_layers = 1
    

    pretrain_count, train_count = 1, 5
    LR = 0.001
    MOMENTUM=0.1
    ACTIVATION = 'softmax'

