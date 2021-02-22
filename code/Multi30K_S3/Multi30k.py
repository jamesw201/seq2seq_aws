import os
import xml.etree.ElementTree as ET
import glob
import io
import codecs
from torchtext import data
from io import TextIOWrapper, BytesIO
import tarfile


class Multi30k(data.Dataset):
    urls = ['https://ml-experiments-20210207.s3-eu-west-1.amazonaws.com/training/training.tar.gz',
              'https://ml-experiments-20210207.s3-eu-west-1.amazonaws.com/testing/validation.tar.gz',
              'https://ml-experiments-20210207.s3-eu-west-1.amazonaws.com/test.tar.gz']
    name = 'multi30k'
    dirname = ''
    directoryname = ''

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, tar_file, path, exts, fields, **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        mtrans = []

        uncompressed_keys=[f'{path}{ext}' for ext in exts]
        print(f'uncompressed_keys: {uncompressed_keys}')

        with tarfile.open(tar_file) as tar:
            source = tar.extractfile(uncompressed_keys[0]).read()
            target = tar.extractfile(uncompressed_keys[1]).read()
            src_lines = source.splitlines()
            trg_lines = target.splitlines()
            print(f'source: {src_lines[0]}')
            print(f'target: {trg_lines[0]}')
            for src_line, trg_line in zip(src_lines, trg_lines):
                src_line, trg_line = str(src_line).strip(), str(trg_line).strip()
                if src_line != '' and trg_line != '':
                    mtrans.append(data.Example.fromlist(
                        [src_line, trg_line], fields))

        super(Multi30k, self).__init__(mtrans, fields, **kwargs)


    @classmethod
    def splits(cls, exts, fields, **kwargs):
        train_path = os.environ['SM_CHANNEL_TRAIN'] + '/' + os.listdir(os.environ['SM_CHANNEL_TRAIN'])[0]
        validation_path = os.environ['SM_CHANNEL_TEST'] + '/' + os.listdir(os.environ['SM_CHANNEL_TEST'])[0]
        evaluation_path = os.environ['SM_CHANNEL_EVAL'] + '/' + os.listdir(os.environ['SM_CHANNEL_EVAL'])[0]

        train_data = cls(train_path, 'train', exts, fields, **kwargs)
        val_data = cls(validation_path, 'val', exts, fields, **kwargs)
        test_data = cls(evaluation_path, 'test2016', exts, fields, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)