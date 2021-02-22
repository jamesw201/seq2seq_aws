import os
from Multi30K_S3.MachineTranslation import MachineTranslation

class Multi30k(MachineTranslation):
    # urls = ['http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/training.tar.gz',
    #         'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz',
    #         'http://www.quest.dcs.shef.ac.uk/'
    #         'wmt17_files_mmt/mmt_task1_test2016.tar.gz']
    
    # Expected url array
    urls = ['https://ml-experiments-20210207.s3-eu-west-1.amazonaws.com/training/training.tar.gz',
              'https://ml-experiments-20210207.s3-eu-west-1.amazonaws.com/testing/validation.tar.gz',
              'https://ml-experiments-20210207.s3-eu-west-1.amazonaws.com/test.tar.gz']
    name = 'multi30k'
    dirname = ''
    directoryname = ''

    @classmethod
    def splits(cls, exts, fields,
               train='training/training.tar.gz', validation='testing/validation.tar.gz', test='test.tar.gz', **kwargs):

        return super(Multi30k, cls).splits(exts, fields, 'ml-experiments-20210207', train, validation, test, **kwargs)