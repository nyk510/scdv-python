# coding: utf-8
"""
"""
import os
import subprocess

from nyktools.utils import download_from_gdrive

__author__ = "nyk510"


def download_ja_w2v_model(to='/data/models/'):
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
    else:
        print('model already exist')

    subprocess.run(['unzip', dl_path, '-d', to])
