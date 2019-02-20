# coding: utf-8
"""ライブドアコーパスから分かち書きされた文章を作成する.
分かち書き結果は pickle して保存されます.
"""
import os
import pickle
from glob import glob

import MeCab

from nyktools import setting
from nyktools.nlp.preprocess import normalize_neologd
from nyktools.nlp.preprocess.models import OchasenLine
from nyktools.nlp.preprocess.wakati import Stopper
from nyktools.utils import get_logger

logger = get_logger(__name__)


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
        lines = self.tagger.parse(s).splitlines()[:-1]
        ocha_lines = [OchasenLine(l) for l in lines]

        return [self.get_word(ocha) for ocha in ocha_lines if self.is_valid_line(ocha)]


def create_parsed_document():
    docs = []

    def trim_top_lines(d):
        return '\n'.join(d.split('\n')[2:])

    for p in glob('/data/livedoor/text/*/*.txt'):
        with open(p) as f:
            s = f.read()
            s = trim_top_lines(s)
            docs.append(s)

    parser = DocumentParser(stopper=Stopper(stop_hinshi='contents'))
    parsed_docs = [parser.call(s) for s in docs]

    # 英数字の lower などやってない. あとで直す
    return parsed_docs


def main():
    parsed = create_parsed_document()

    save_path = os.path.join(setting.FEATURE_DIR, 'parsed_docs.pkl')
    with open(save_path, mode='wb') as f:
        logger.info('save to {}'.format(save_path))
        pickle.dump(parsed, f)


if __name__ == '__main__':
    main()
