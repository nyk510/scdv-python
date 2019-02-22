#!/usr/bin/env python
# coding: utf-8

import os
import pickle
from collections import OrderedDict
from glob import glob

import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from nyktools import setting
from nyktools.nlp.models import ja_w2v_model
from nyktools.utils import get_logger, set_default_style, stopwatch

logger = get_logger(__name__)


def load_parsed_document():
    with open(os.path.join(setting.FEATURE_DIR, 'parsed_docs.pkl'), 'rb') as f:
        doc = pickle.load(f)
    return doc


w2v_model = ja_w2v_model()


def convert_to_wv(w):
    """
    単語から word2vec の特徴量に変換する.
    単語が登録されていない (vocabularyにない) ときは zero vector を返す
    
    args:
        w(str): 変換したい word
    """
    try:
        v = w2v_model.word_vec(w)
    except KeyError as e:
        #         logger.warning(e)
        v = np.zeros(shape=(w2v_model.vector_size,))
    return v


@stopwatch
def create_swem(doc, aggregation='max'):
    """
    Create Simple Word Embedding Model Vector from document (i.e. list of sentence)
    Args:
        doc(list[list[str]]):
        aggregation(str): `"max"` or `"mean"`

    Returns:

    """
    print('create SWEM: {}'.format(aggregation))
    if aggregation == 'max':
        agg = np.max
    elif aggregation == 'mean':
        agg = np.mean
    else:
        raise ValueError()

    swem = []
    for sentence in tqdm(doc, total=len(doc)):
        embed_i = [convert_to_wv(s) for s in sentence]
        embed_i = np.array(embed_i)

        # max-pooling で各文章を 300 次元のベクトルに圧縮する
        embed_i = agg(embed_i, axis=0)
        swem.append(embed_i)
    swem = np.array(swem)
    return swem


# ## N-Gram SWEM
# 
# n-gram の平均ベクトルを用いて max-pooling する swem の発展形
def create_n_gram_max_pooling(s, n_gram_length=5):
    """
    単語に区切られた文字列から n_gram の max-pooling vector を返す
    
    Returns:
        np.ndarray: shape = (n_embedded_dim, )
    """
    embed_i = [convert_to_wv(w) for w in s]
    gram_vectors = []
    for i in range(max(1, len(embed_i) - n_gram_length)):
        gram_i = embed_i[i:i + n_gram_length]
        gram_mean = np.mean(gram_i, axis=0)
        gram_vectors.append(gram_mean)
    return np.max(gram_vectors, axis=0)


def create_n_gram_feature(docs, n_gram=5):
    vectors = []
    for sentence in tqdm(docs, total=len(docs)):
        v = create_n_gram_max_pooling(sentence, n_gram)
        vectors.append(v)
    return np.array(vectors)


def train_lgbm(feature, categorical_target):
    """
    LightGBM Classifier による学習を実行する

    Args:
        feature:
        categorical_target:

    Returns:

    """
    n_cv = 5
    fold = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=19)
    y_pred = np.zeros_like(categorical_target, dtype=np.int32)
    cv_num = np.zeros_like(categorical_target, dtype=np.int32)

    for i, (idx_train, idx_valid) in enumerate(fold.split(feature, categorical_target)):
        print('cv:{}/{}'.format(i + 1, n_cv))
        x_train, y_train = feature[idx_train], categorical_target[idx_train]
        x_valid, y_valid = feature[idx_valid], categorical_target[idx_valid]
        clf = lgbm.LGBMClassifier(n_estimators=300)
        clf.fit(x_train, y_train,
                eval_set=[(x_valid, y_valid)], early_stopping_rounds=40,
                verbose=40, eval_metric=['multi_error'])
        y_pred_i = clf.predict(x_valid)
        y_pred[idx_valid] = y_pred_i
        cv_num[idx_valid] = i

    df_pred = pd.DataFrame({
        'predict': y_pred,
        'cv': cv_num
    })

    acc_score = accuracy_score(categorical_target, df_pred.predict)
    print('accuracy: {:.4f}'.format(acc_score))

    return df_pred


def main():
    label_encoder = LabelEncoder()

    target = []
    for p in glob('/data/livedoor/text/*/*.txt'):
        target.append(p.split('/')[-2])
    target = np.array(target)

    categorical_target = label_encoder.fit_transform(target)

    # 特徴量を集める
    use_dataset = OrderedDict()
    parsed_doc = load_parsed_document()

    # Simple Word Embedding Model
    for agg in ['mean', 'max']:
        use_dataset['swem_{}'.format(agg)] = create_swem(parsed_doc, agg)

    # N-Gram SWEM
    for n in [3, 5, 8]:
        use_dataset['n={}_gram_swem'.format(n)] = create_n_gram_feature(parsed_doc, n)

    # SCDV
    scdv = np.load('/data/processed/compressed_document_vector.npy')
    scdv_raw = np.load('/data/processed/raw_document_vector.npy')
    use_dataset['scdv'] = scdv
    use_dataset['scdv_raw'] = scdv_raw

    # 次元数めっちゃ多いので PCA で圧縮したものも使ってみる

    hidden_dims = [100, 300, 500]

    for h in hidden_dims:
        pca_clf = PCA(n_components=h)
        use_dataset['scda_pca_{}'.format(h)] = pca_clf.fit_transform(scdv)

    # ## Training
    #
    # * それぞれの特徴量に対して 5 fold CV を実行して, accuracy を見る

    output_dir = '/data/visualize'
    os.makedirs(output_dir, exist_ok=True)

    df_pred = None
    for name, feat in use_dataset.items():
        logger.info('start {}'.format(name))
        df_i = train_lgbm(feat, categorical_target)
        df_i['model_name'] = name

        if df_pred is None:
            df_pred = df_i
        else:
            df_pred = pd.concat([df_pred, df_i], ignore_index=True)

    # 答え合わせ
    score_data = []

    for m in df_pred.model_name.unique():
        _df = df_pred[df_pred.model_name == m]

        for n in _df.cv.unique():
            idx = _df.cv == n
            t = categorical_target[idx]
            pred = _df[idx].predict

            score_data.append({'cv': n, 'accuracy': accuracy_score(t, pred), 'feature': m})

    df_score = pd.DataFrame(score_data)
    df_score.to_csv(os.path.join(output_dir, 'score.csv'))

    set_default_style()

    order = df_score.groupby('feature').mean().sort_values('accuracy', ascending=False).index.values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax = sns.violinplot(data=df_score, x='feature', y='accuracy', ax=ax, order=order)
    ax = sns.stripplot(data=df_score, x='feature', y='accuracy', ax=ax, color='grey', order=order)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'swem_vs_scdv.png'), dpi=120)


if __name__ == '__main__':
    main()
