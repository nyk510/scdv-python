# coding: utf-8
"""
日本語の文章からMeCabを用いて
分かち書きされたトークンを作成するモジュール
"""

__author__ = "yamaguchi"

import MeCab

from .constants import HINSHI_FOR_CONTENTS, WORDS_FOR_CONTENTS
from .models import OchasenLine
from .normalize import normalize_neologd


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


class DocumentParser(object):
    def __init__(self, stopper=None, as_normed=True):
        """

        Args:
            stopper(Stopper | None):
                stop word を拡張した Stopper instance.
                分かち書き結果に含めたくない単語などが有る場合, stopper クラスを渡す.
                特に指定がない場合すべての単語を分かち書き結果に含める.
            as_normed(bool):
                True のとき原型を分かち書きとして返す.
                文章の意味解析の場合活用はあまり考慮する必要が無いため指定すると吉

        Examples:
            In [2]: parser = DocumentParser()

            In [3]: s = '買えそう'

            In [4]: parser.call(s)
            Out[4]: ['買える', 'そう']
        """

        self.tagger = MeCab.Tagger('-Ochasen')

        if stopper is None:
            stopper = Stopper()
        self.stopper = stopper
        self.as_normed = as_normed

    def get_word(self, ocha):
        """
        Ochasen でわけられた OchasenLine から単語を取得する

        Args:
            ocha(OchasenLine):

        Returns(str):

        """
        if self.as_normed:
            return ocha.norm_word
        else:
            return ocha.word

    def is_valid_line(self, ocha):
        """

        Args:
            ocha(OchasenLine):

        Returns(bool):

        """
        if self.stopper is None:
            return ocha.can_parse

        return ocha.can_parse and self.stopper(ocha.norm_word, ocha.hinshi_class)

    def call(self, sentence):
        """
        文章の文字列を受け取り分かち書きされた list を返す

        Args:
            sentence(str):

        Returns:
            list[str]
        """
        s = normalize_neologd(sentence)
        # 文字列への事前処理を pytorch の compose っぽく書きたい
        s = s.lower()
        lines = self.tagger.parse(s).splitlines()[:-1]
        ocha_lines = [OchasenLine(l) for l in lines]

        return [self.get_word(ocha) for ocha in ocha_lines if self.is_valid_line(ocha)]
