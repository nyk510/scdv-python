# coding: utf-8
"""
"""
import os
import tarfile
from glob import glob

import requests

from nyktools.setting import DATASET_DIR
from nyktools.utils import save_response_content

__author__ = "nyk510"


class DataSet(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.root_path = os.path.join(DATASET_DIR, self.dirname)
        os.makedirs(self.root_path, exist_ok=True)

    def load(self):
        raise NotImplementedError()


class LivedoorCorpusDataSet(DataSet):
    def __init__(self):
        super(LivedoorCorpusDataSet, self).__init__('livedoor')

        self.raw_path = os.path.join(self.root_path, 'ldcc-20140209.tar.gz')

    def download(self):
        url = 'https://www.rondhuit.com/download/ldcc-20140209.tar.gz'

        save_response_content(requests.get(url), destination=self.raw_path)

        with tarfile.open(self.raw_path) as f:
            f.extractall(self.root_path)

    @property
    def already_exist(self):
        return os.path.exists(self.raw_path)

    def text_paths(self):
        return glob(os.path.join(self.root_path, 'text/*/*.txt'))

    def load(self):
        if not self.already_exist:
            self.download()

        docs = []
        labels = []

        # 先頭行2行はメターデータなので飛ばす
        def trim_top_lines(d):
            return '\n'.join(d.split('\n')[2:])

        for p in self.text_paths():
            with open(p) as f:
                s = f.read()
                s = trim_top_lines(s)
                docs.append(s)

            labels.append(p.split('/')[-2])
        return docs, labels


def livedoor_news():
    return LivedoorCorpusDataSet().load()
