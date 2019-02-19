# coding: utf-8
"""
"""
import os
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from nyktools import setting
from nyktools.nlp.models import ja_w2v_model
from nyktools.utils import get_logger

__author__ = "nyk510"

logger = get_logger(__name__)


def load_parsed_document():
    with open(os.path.join(setting.FEATURE_DIR, 'parsed_docs.pkl'), 'rb') as f:
        doc = pickle.load(f)
    return doc


def create_idf_dataframe(documents, parsed=True):
    """

    Args:
        documents(List[List[str]] | List[str]):
        parsed(bool):

    Returns(pd.DataFrame):

    """
    transformer = TfidfVectorizer()

    if parsed:
        raw_docs = [' '.join(d) for d in documents]
    else:
        raw_docs = documents

    transformer.fit(raw_docs)
    df_idf = pd.DataFrame()
    df_idf['word'] = transformer.get_feature_names()
    df_idf['idf'] = transformer.idf_
    return df_idf


def create_document_vector(documents, word_vocab, word_topic_vector):
    """
    学習済みの word topic vector と分かち書き済みの文章, 使用されている単語から
    文章ベクトルを作成するメソッド.

    word_vocab は word topic vector と対応関係に有るひつようがあります.
    すなわち i 番目の word_vocab に対応するベクトルが word_topic_vector[i] に相当します.

    Args:
        documents(list[list[str]]):
        word_vocab(list[str]):
        word_topic_vector(np.ndarray):

    Returns:

    """
    assert len(word_vocab) == word_topic_vector.shape[0]

    def create_doc_vector(doc):
        vec = np.zeros_like(word_topic_vector[0])

        for w in doc:
            try:
                idx = word_vocab.index(w)
                v_i = word_topic_vector[idx]
                vec += v_i
            except Exception:
                continue
        return vec

    doc_vecs = []
    for doc in tqdm(documents, total=len(documents)):
        doc_vecs.append(create_doc_vector(doc))
    return np.array(doc_vecs)


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
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--components', default=60, type=int, help='GMM component size (i.e. latent space size.)')
    return parser.parse_args()


def main():
    args = vars(get_arguments())

    w2v_model = ja_w2v_model()

    output_dir = os.path.join(setting.PROCESSED_ROOT)
    n_wv_embed = w2v_model.vector_size
    n_components = args['components']
    logger.info('w2v embed:{}\tGMM components:{}'.format(n_wv_embed, n_components))

    parsed_docs = load_parsed_document()
    vocab_model = set(k for k in w2v_model.vocab.keys())
    vocab_docs = set([w for doc in parsed_docs for w in doc])
    out_of_vocabs = len(vocab_docs) - len(vocab_docs & vocab_model)
    print('out of vocabs: {out_of_vocabs}'.format(**locals()))

    # 使う文章に入っているものだけ学習させるため共通集合を取得してその word vector を GMM の入力にする
    use_words = list(vocab_docs & vocab_model)

    # 使う単語分だけ word vector を取得. よって shape = (n_vocabs, n_wv_embed,)
    use_word_vectors = np.array([w2v_model[w] for w in use_words])

    # 公式実装: https://github.com/dheeraj7596/SCDV/blob/master/20news/SCDV.py#L32 により tied で学習
    # 共分散行列全部推定する必要が有るほど低次元ではないという判断?
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
    df_idf = create_idf_dataframe(parsed_docs, parsed=True)
    df_use = pd.merge(df_use, df_idf, on='word', how='left')
    idf = df_use['idf'].values

    # NOTE: この merge 後の dataframe に idf = np.nan のデータがいくつか含まれる(だいたい5000ぐらい)
    # use_word は parsed_doc に現れる word の一部なのでこれはおかしい. 調査が必要.
    # tfidf transformer の使い方があやしい
    df_use[df_use.isnull()].to_csv(os.path.join(setting.PROCESSED_ROOT, 'idf_has_null.csv'))

    # topic vector を計算するときに concatenation するとあるが
    # 単に 二次元のベクトルに変形して各 vocab に対して idf をかければ OK
    word_topic_vector = word_cluster_vector.reshape(-1, n_components * n_wv_embed) * idf[:, None]

    # nan になっている部分は idf 計算で nan が出現したもの
    # 一旦影響が出ないように 0 で埋める
    word_topic_vector[np.isnan(word_topic_vector)] = 0

    np.save(os.path.join(output_dir, 'word_topic_vector.npy'), word_topic_vector)

    word_topic_vector = np.load(os.path.join(output_dir, 'word_topic_vector.npy'))

    document_vector = create_document_vector(parsed_docs, use_words, word_topic_vector)
    np.save(os.path.join(output_dir, 'raw_document_vector.npy'), document_vector)

    compressed = compress_document_vector(document_vector)
    np.save(os.path.join(output_dir, 'compressed_document_vector.npy'), compressed)


if __name__ == '__main__':
    main()
