# coding: utf-8
"""訓練済みモデルを定義するファイル
"""
import os
import zipfile

import gensim

from nyktools.setting import DATA_DIR
from nyktools.utils import download_from_gdrive, get_logger

__author__ = "nyk510"

logger = get_logger(__name__)


def ja_word_vector(to='models/'):
    """
    fasttext を用いて学習された日本語の word vector を取得します

    データは以下の記事にあるものを使わせてもらっています. 感謝してつかいましょう^_^
    > https://qiita.com/Hironsan/items/513b9f93752ecee9e670 

    Args:
        to: 保存先ディレクトリ

    Returns:

    """
    to = os.path.join(DATA_DIR, to)
    file_name = 'vector_neologd'
    dl_path = os.path.join(to, '{}.zip'.format(file_name))
    # 展開すると model.vec という名前のファイルがあるのでそれが本体
    model_path = os.path.join(to, 'model.vec')

    if not os.path.exists(model_path):
        os.makedirs(to, exist_ok=True)
        download_from_gdrive('0ByFQ96A4DgSPUm9wVWRLdm5qbmc', destination=dl_path)
        with zipfile.ZipFile(dl_path) as f:
            f.extractall(to)
    else:
        print('model already exist')

    logger.info('start loading W2V Model...')

    m = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
    logger.info('finished')
    return m
