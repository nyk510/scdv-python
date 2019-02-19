# coding: utf-8
"""
ストップワード, 品詞の条件に基づいて
単語をコーパスに載せるかを判断するstoperの定義
"""

__author__ = "yamaguchi"


class OchasenLine(object):
    def __init__(self, line):
        x = line.split('\t')
        try:
            # 単語そのもの: かかわり
            self.word = x[0]
            # 単語の読み方: カカワリ
            self.yomi = x[1]
            # 原型 かかわり -> かかわる
            self.norm_word = x[2]

            # 推定される品詞: 助詞-格助詞-一般
            hinshi = x[3].split('-')
            self.hinshi_class = hinshi[0]

            if len(hinshi) > 1:
                self.hinshi_detail = hinshi[1]
            else:
                self.hinshi_detail = None

            self.can_parse = True
        except Exception as e:
            self.can_parse = False
            print('not parse: {}'.format(line), e)
        self.line = line

    def __str__(self):
        if self.can_parse:
            return '{0.word}-{0.yomi}'.format(self)
        return None

    def __repr__(self):
        return self.__str__()
