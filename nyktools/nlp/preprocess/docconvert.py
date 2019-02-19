# coding: utf-8
"""
文章からLDA, Word2Vec等の分かち書きの特徴量を受け取るモデルに渡すことのできるデータを作成
"""

import os
import pickle

from gensim import corpora

from .wakati import Wakati


class DocConverter(object):
    """
    文章を分かち書き特徴量に変換
    """

    def __init__(self, documents, wakati, project_name, save_to=None):
        """
        Args:
            parsed_docs (list)
                分かち書きされている文章を返すリストもしくはイテレータ
            project_name (String)
                ファイルは project_name.~ の形式で INPUTDIR に保存されます
        """

        self.documents = documents
        if isinstance(wakati, Wakati) is False:
            raise TypeError("wakati is not a instance of Wakati.")
        self.project_name = project_name
        self.save_to = save_to

        if self.save_to is not None:
            os.makedirs(self.save_to, exist_ok=True)

    @property
    def run_save(self):
        return self.save_to is not None

    def run(self, no_below=5, no_above=0.5, keep_n=10000):
        """
        文章の変換

        Args:
            min_count (int)
                設定された回数以下しか出現しないトークンを取り除きます.
                (ちょうどmin_count回だけ登場するトークンは残ります)
        """
        docs = self.documents
        p_name = self.project_name

        # 分かち書き文書から単語の辞書を作成
        doc_dict = corpora.Dictionary(docs)
        if self.run_save:
            doc_dict.save_as_text(
                os.path.join(self.save_to, "{0}_dict_beforefiltered.txt".format(p_name)))

        # 指定された条件で絞り込み
        doc_dict.filter_extremes(no_below, no_above, keep_n)
        if self.run_save:
            doc_dict.save(os.path.join(self.save_to, "{0}.dict".format(p_name)))

        # BoWとDictからCorpusを作成
        corpus = [doc_dict.doc2bow(doc) for doc in docs]

        if self.run_save:
            f_name = os.path.join(self.save_to, "{0}.corpus".format(p_name))
            corpora.MmCorpus.serialize(f_name, corpus)

        # tokenをidに変換したドキュメント
        doc_transformed_to_id = [[doc_dict.token2id[token] for token in doc]
                                 for doc in docs]
        f_name = os.path.join(self.save_to,
                           "{0}.doc_transported2id.bin".format(p_name))
        with open(f_name, "wb") as f:
            pickle.dump(doc_transformed_to_id, f)

        f_name = os.path.join(self.save_to, "{0}.word_frequency.txt".format(p_name))
        with open(f_name, "w") as f:
            for key, val in freq_dict.items():
                f.write("{key}\t{val}\n".format(**locals()))
        return
