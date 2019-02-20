# coding: utf-8
"""
"""

from logging import getLogger, StreamHandler, FileHandler, Formatter
from time import time

import numpy as np
import requests
import seaborn as sns
from tqdm import tqdm


def set_default_style(style='ticks', font='Noto Sans CJK JP'):
    """
    matplotlib, seaborn でのグラフ描写スタイルを標準的仕様に設定するメソッド
    このメソッドの呼び出しは破壊的です。

    Args:
        style(str):
        font(str):

    Returns: None

    """
    sns.set(style=style, font=font)


def get_logger(name, log_level="DEBUG", output_file=None, handler_level="INFO"):
    """
    :param str name:
    :param str log_level:
    :param str | None output_file:
    :return: logger
    """
    logger = getLogger(name)

    formatter = Formatter("[%(asctime)s] %(message)s")

    handler = StreamHandler()
    logger.setLevel(log_level)
    handler.setLevel(handler_level)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if output_file:
        file_handler = FileHandler(output_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(handler_level)
        logger.addHandler(file_handler)

    return logger


def get_sample_pos_weight(y):
    """

    Args:
        y(np.ndarray): shape = (n_samples, )

    Returns:
        float
    """
    unique, count = np.unique(y, return_counts=True)
    y_sample_weight = dict(zip(unique, count))
    sample_pos_weight = y_sample_weight[0] / y_sample_weight[1]
    return sample_pos_weight


logger = get_logger(__name__)


def download_from_gdrive(id, destination):
    """
    Download file from Google Drive
    :param str id: g-drive id
    :param str destination: output path
    :return:
    """
    url = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(url, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        logger.info("get download warning. set confirm token.")
        params = {'id': id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    """
    verify whether warned or not.

    [note] In Google Drive Api, if requests content size is large,
    the user are send to verification page.

    :param requests.Response response:
    :return:
    """
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            return v

    return None


def save_response_content(response, destination):
    """
    :param requests.Response response:
    :param str destination:
    :return:
    """
    chunk_size = 1024 * 1024
    logger.info("start downloading...")
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), unit="MB"):
            f.write(chunk)
    logger.info("Finish!!")
    logger.info("Save to:{}".format(destination))


def stopwatch(func):
    """
    実行時間を計測する decorator
    Args:
        func:

    Returns:

    """

    def inner(*args, **kwargs):
        start = time()
        res = func(*args, **kwargs)
        diff = time() - start
        print('[{}] time: {:.3f}[s]'.format(func.__name__, diff))
        return res

    return inner
