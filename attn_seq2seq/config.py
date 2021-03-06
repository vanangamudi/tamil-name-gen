import logging
from hpconfig import CONFIG as HPCONFIG

#FORMAT_STRING = "%(levelname)-8s:%(name)s.%(funcName)s>> %(message)s"
FORMAT_STRING = "%(levelname)-8s:%(name)-8s.%(funcName)-8ss>> %(message)s"

class ConfigMeta(type):
    def __getattr__(cls, attr):
        return cls._default

class Base(metaclass=ConfigMeta):
    pass

class CONFIG(Base):
    split_ratio = 1.0
    dropout = 0.1
    cuda = True
    multi_gpu = False
    tqdm = True
    flush = False
    batch_size = 3000
    plot_metrics = True


    write_vocab_to_file = True
    dump_bloom_filter = False
    
    CHECKPOINT = 1
    EPOCHS = 2000
    EONS=4
    #TIMES=10
    ACCURACY_THRESHOLD=0.9
    ACCURACY_IMPROVEMENT_THRESHOLD=0.05
    
    class LOG(Base):
        class _default(Base):
            level=logging.CRITICAL
        class PREPROCESS(Base):
            level=logging.DEBUG
        class MODEL(Base):
            level=logging.INFO
        class TRAINER(Base):
            level=logging.INFO
        class DATAFEED(Base):
            level=logging.INFO
