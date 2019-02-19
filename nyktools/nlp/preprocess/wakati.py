# coding: utf-8
"""
日本語の文章からMeCabを用いて
分かち書きされたトークンを作成するモジュール
"""

__author__ = "yamaguchi"

import re

import MeCab

from .constants import HINSHI_FOR_CONTENTS, WORDS_FOR_CONTENTS, HANKAKU_PATTARN


class Stopper(object):
    """
    単語とクラスを受取り, 単語を残すかどうかを判定
    ストップワードや、特定の品詞を取り除くことを想定しています。
    """

    def __init__(self,
                 stop_hinshi=None,
                 stop_words=None,
                 remove_sign=True,
                 remove_oneword=True):
        """
        Args:
            stop_hinshi (dictionary | str)
                取り除く品詞の入ったディクショナリ。
                Noneのときはデフォルト値として空のリスト`[]`を用います.
            stop_words (List<string> | str)
                ストップワードのリスト.
                Noneのときはデフォルトとして空のリスト`[]`を用います.
            remove_sign (boolean)
                記号を除去するかどうかのboolean. デフォルト値はTrue.
            remove_oneword (boolean)
                一文字のwordを取り除くかどうかのboolean.
                機械翻訳がタスクの場合、係り受けを考慮しなくてはならないので, 一字であっても除去しないのが普通です。
                文章の内容の要約がタスクの場合、一字の言葉は内容を表していない場合が多くあるという観点から除去する場合があります。
                要するにタスク依存です。これは記号除去にも同じことがいえます。
        """

        if stop_hinshi is None:
            self.stop_hinshi = {}
        elif stop_hinshi == "contents":
            self.stop_hinshi = HINSHI_FOR_CONTENTS
        else:
            self.stop_hinshi = stop_hinshi

        if stop_words is None:
            self.stop_words = []
        elif stop_words == "contents":
            self.stop_words = WORDS_FOR_CONTENTS
        else:
            self.stop_words = stop_words

        self.remove_sign = remove_sign
        self.remove_oneword = remove_oneword

    def is_oneword(self, word):
        """
        単語が一文字であるかどうか判定

        Args:
            word (string)

        Returns:
            word is constructed with one word or not (Boolean)
        """

        if len(word) == 1:
            return True
        else:
            return False

    def __call__(self, word, word_class):
        """

        Args:
            word:
            word_class:

        Returns:
            boolean
        """

        if self.remove_oneword and self.is_oneword(word):
            return False

        if word in self.stop_words:
            return False

        for key, value in self.stop_hinshi.items():
            for w_class in word_class:
                if w_class in value:
                    return False
        return True


class Wakati(object):
    """
    MeCabを用いて分かち書きを行うクラス.
    """

    def __init__(self,
                 stoper,
                 tagger_type='-Ochasen',
                 do_remove_hankakusign=True):
        """
        分かち書きを行うクラスのインスタンス

        Args:
            stoper (instance of Discriminator)
            tagger_type (args of mecab tagger)
                `MeCab.Tager`インスタンス作成時の引数
            do_remove_hankaku_signs (boolean)
                半角記号を取り除くフラグ
        """

        self.stoper = stoper
        self.do_remove_hankakusign = do_remove_hankakusign
        self.tagger_type = tagger_type
        self.tagger = MeCab.Tagger(tagger_type)

        # python3のmecabはそのままparseを呼び出すとはじめ空文字をかえすバグが有る
        # それを修正するための慣用句
        self.tagger.parse("")

    def _remove_hankaku_sign(self, sentence):
        """
        文章中の記号を除去
        todo: 今は半角記号しか対応できていないので直す

        Args:
            sentence (string)

        Returns:
            string
        """

        subed_sentence = re.sub(HANKAKU_PATTARN, "", sentence)
        return subed_sentence

    def parse(self, sentence):
        """
        文章を単語(token)のリストに分解

        Args:
            sentence (string)

        Returns:
            list of words
        """
        return self.parse2word_and_class(sentence, hinshi_depth=0)

    def parse2word_and_class(self, sentence, hinshi_depth=2):
        """
        文章を単語(word)と品詞(word_class)のタプルのリストに分解します。

        Args:
            sentence (string)
                分かち書きする文章
            stoper (a Discriminator class instance)
            hinshi_depth (int)
                品詞分解をどれだけ深く探索するか。
                デフォルト値の2では二段階の探索を行い, 品詞, 品詞小分類1をword_classとして返す

        Returns:
            list of tupple [word, word_class].
        """
        if self.do_remove_hankakusign:
            sentence = self._remove_hankaku_sign(sentence)

        parsed_doc = []
        node = self.tagger.parseToNode(sentence)
        while node:
            if node.prev is None:
                node = node.next
                continue
            before_s = node.prev.surface
            after_s = node.surface
            word = before_s[:-len(after_s)]
            wclass = node.feature.split(",")
            if wclass[0] != "BOS/EOS" and self.stoper(word, wclass):
                if hinshi_depth == 0:
                    parsed_doc.append(word)
                else:
                    parsed_doc.append([word, wclass[0:hinshi_depth]])
            node = node.next
        return parsed_doc
