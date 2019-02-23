# coding: utf-8
"""Create SCDV using livedoor corpus dataset
"""
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from nyktools import setting
from nyktools.nlp.applications import ja_word_vector
from nyktools.nlp.dataset import livedoor_news
from nyktools.nlp.preprocess import Stopper, DocumentParser
from nyktools.utils import get_logger, stopwatch

__author__ = "nyk510"

logger = get_logger(__name__)


@stopwatch
def create_parsed_document(docs):
    parser = DocumentParser(stopper=Stopper(stop_hinshi='contents'))
    parsed_docs = [parser.call(s) for s in docs]
    return parsed_docs


def create_idf_dataframe(documents):
    """

    Args:
        documents(list[str]):
    Returns(pd.DataFrame):

    """

    d = defaultdict(int)

    for doc in documents:
        vocab_i = set(doc)
        for w in list(vocab_i):
            d[w] += 1

    df_idf = pd.DataFrame()
    df_idf['count'] = d.values()
    df_idf['word'] = d.keys()
    df_idf['idf'] = np.log(len(documents) / df_idf['count'])
    return df_idf


def create_document_vector(documents, w2t, n_embedding):
    """
    学習済みの word topic vector と分かち書き済みの文章, 使用されている単語から
    文章ベクトルを作成するメソッド.

    Args:
        documents(list[list[str]]):
        w2t(dict): 単語 -> 埋め込み次元の dict
        n_embedding(int):

    Returns:
        embedded document vector

    """
    doc_vectors = []

    for doc in documents:
        vector_i = np.zeros(shape=(n_embedding,))
        for w in doc:
            try:
                v = w2t[w]
                vector_i += v
            except KeyError:
                continue
        doc_vectors.append(vector_i)
    return np.array(doc_vectors)


def compress_document_vector(doc_vector, p=.04):
    v = np.copy(doc_vector)
    vec_norm = np.linalg.norm(v, axis=1)
    # zero divide しないように
    vec_norm = np.where(vec_norm > 0, vec_norm, 1.)
    v /= vec_norm[:, None]

    a_min = v.min(axis=1).mean()
    a_max = v.max(axis=1).mean()
    threshold = (abs(a_min) + abs(a_max)) / 2. * p
    v[abs(v) < threshold] = .0
    return v


def get_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, usage=__doc__)
    parser.add_argument('-c', '--components', default=60, type=int, help='GMM component size (i.e. latent space size.)')
    return parser.parse_args()


def main():
    args = vars(get_arguments())

    word_vec = ja_word_vector()

    output_dir = os.path.join(setting.PROCESSED_ROOT)
    n_wv_embed = word_vec.vector_size
    n_components = args['components']
    logger.info('w2v embed:{}\tGMM components:{}'.format(n_wv_embed, n_components))

    docs, _ = livedoor_news()
    parsed_docs = create_parsed_document(docs)

    # w2v model と corpus の語彙集合を作成
    vocab_model = set(k for k in word_vec.vocab.keys())
    vocab_docs = set([w for doc in parsed_docs for w in doc])
    out_of_vocabs = len(vocab_docs) - len(vocab_docs & vocab_model)
    print('out of vocabs: {out_of_vocabs}'.format(**locals()))

    # 使う文章に入っているものだけ学習させるため共通集合を取得してその word vector を GMM の入力にする
    use_words = list(vocab_docs & vocab_model)

    # 使う単語分だけ word vector を取得. よって shape = (n_vocabs, n_wv_embed,)
    use_word_vectors = np.array([word_vec[w] for w in use_words])

    # 公式実装: https://github.com/dheeraj7596/SCDV/blob/master/20news/SCDV.py#L32 により tied で学習
    # 共分散行列全部推定する必要が有るほど低次元ではないという判断?
    # -> 多分各クラスの分散を共通化することで各クラスに所属するデータ数を揃えたいとうのがお気持ちっぽい
    clf = GaussianMixture(n_components=n_components, covariance_type='tied', verbose=2)
    clf.fit(use_word_vectors)

    # word probs は各単語のクラスタへの割当確率なので shape = (n_vocabs, n_components,)
    word_probs = clf.predict_proba(use_word_vectors)

    # 単語ごとにクラスタへの割当確率を wv に対して掛け算する
    # shape = (n_vocabs, n_components, n_wv_embed) になる
    word_cluster_vector = use_word_vectors[:, None, :] * word_probs[:, :, None]

    # はじめに文章全体の idf を作成した後, use_word だけの df と left join して
    # 使用している単語の idf を取得
    df_use = pd.DataFrame()
    df_use['word'] = use_words
    df_idf = create_idf_dataframe(parsed_docs)
    df_use = pd.merge(df_use, df_idf, on='word', how='left')
    idf = df_use['idf'].values

    # topic vector を計算するときに concatenation するとあるが
    # 単に 二次元のベクトルに変形して各 vocab に対して idf をかければ OK
    topic_vector = word_cluster_vector.reshape(-1, n_components * n_wv_embed) * idf[:, None]
    # nanで影響が出ないように 0 で埋める
    topic_vector[np.isnan(topic_vector)] = 0
    word_to_topic = dict(zip(use_words, topic_vector))

    np.save(os.path.join(output_dir, 'word_topic_vector.npy'), topic_vector)

    topic_vector = np.load(os.path.join(output_dir, 'word_topic_vector.npy'))
    n_embedding = topic_vector.shape[1]

    cdv_vector = create_document_vector(parsed_docs, word_to_topic, n_embedding)
    np.save(os.path.join(output_dir, 'raw_document_vector.npy'), cdv_vector)

    compressed = compress_document_vector(cdv_vector)
    np.save(os.path.join(output_dir, 'compressed_document_vector.npy'), compressed)


if __name__ == '__main__':
    main()
