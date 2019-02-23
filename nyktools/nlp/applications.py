# coding: utf-8
"""
"""
import os
import zipfile

import gensim

from nyktools.utils import download_from_gdrive, get_logger

__author__ = "nyk510"

logger = get_logger(__name__)


def ja_word_vector(to='/data/models/'):
    """
    日本語の訓練済み Word2Vec モデルを download する
    Args:
        to:

    Returns:

    """
    file_name = 'vector_neologd'
    dl_path = os.path.join(to, '{}.zip'.format(file_name))

    if not os.path.exists(dl_path):
        os.makedirs(to, exist_ok=True)
        download_from_gdrive('0ByFQ96A4DgSPUm9wVWRLdm5qbmc', destination=dl_path)
    else:
        print('model already exist')
    with zipfile.ZipFile(dl_path) as f:
        f.extractall(to)

    logger.info('start loading W2V Model...')
    model_path = os.path.join(to, 'model.vec')
    m = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
    logger.info('finished')
    return m
