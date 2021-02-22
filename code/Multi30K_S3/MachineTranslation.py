import os
import xml.etree.ElementTree as ET
import glob
import io
import codecs
from torchtext import data
import io
import boto3
from io import TextIOWrapper, BytesIO
from gzip import GzipFile
import tarfile

session = boto3.Session()
s3 = session.client('s3')


class MachineTranslation(data.Dataset):
    
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, bucket, key, path, exts, fields, **kwargs):
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]

        mtrans = []
        
        s3_resource = boto3.resource('s3')
        print(f'bucket: {bucket}, key: {key}')

#         print(f'file location: {os.environ[key]}')
    
        gzipped_archive = s3_resource.Object(bucket_name=bucket, key=key)
        input_tar_content = gzipped_archive.get()['Body'].read()

        uncompressed_keys=[f'{path}{ext}' for ext in exts]
        print(f'uncompressed_keys: {uncompressed_keys}')
        with tarfile.open(fileobj = BytesIO(input_tar_content)) as tar:
#         with tarfile.open(os.environ[key]) as tar:
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

        super(MachineTranslation, self).__init__(mtrans, fields, **kwargs)


    @classmethod
    def splits(cls, exts, fields, bucket=None, 
               train='training/training.tar.gz', validation='testing/validation.tar.gz', test='test.tar.gz', **kwargs):
        print(f'class name: {cls.__name__}')
        if bucket is None:
            raise Exception('a bucket must be specified')
        train_data = None if train is None else cls(bucket, train, 'train', exts, fields, **kwargs)
        val_data = None if validation is None else cls(bucket, validation, 'val', exts, fields, **kwargs)
        test_data = None if test is None else cls(bucket, test, 'test2016', exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)