# coding: utf-8
"""
"""
import os
import subprocess

import gensim

from nyktools.utils import download_from_gdrive, get_logger

__author__ = "nyk510"

logger = get_logger(__name__)


def ja_w2v_model(to='/data/models/'):
    """
    日本語の訓練済み Word2Vec モデルを download する
    Args:
        to:

    Returns:

    """
    file_name = 'vector_neologd'
    dl_path = os.path.join(to, '{}.zip'.format(file_name))

    if not os.path.exists(dl_path):
        download_from_gdrive('0ByFQ96A4DgSPUm9wVWRLdm5qbmc', destination=dl_path)
        subprocess.run(['unzip', dl_path, '-d', to, '-y'])
    else:
        print('model already exist')

    logger.info('start loading W2V Model...')
    model_path = os.path.join(to, 'model.vec')
    m = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
    logger.info('finished')
    return m
