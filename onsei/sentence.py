from dataclasses import dataclass
from typing import Optional, List, Tuple

import MeCab
import jaconv

from furigana.furigana import Text, parse_node, is_katakana


class Sentence:
    def __init__(self, raw_sentence: str):
        self.raw_sentence = raw_sentence.strip()

        self.words = parse_words(raw_sentence)

        self.julius_transcript = generate_julius_transcript_from_words(self.words)

    def to_html(self):
        return ' '.join((word.to_html() for word in self.words))


def parse_words(sentence: str) -> List['Word']:
    mecab = MeCab.Tagger("-Ochasen")
    mecab.parse('')  # 空でパースする必要がある
    node = mecab.parseToNode(sentence)
    words = []

    while node is not None:
        raw = node.surface
        features = node.feature.split(",")
        pos_type = features[0]

        phonetics = ruby = None
        if pos_type != u"記号":
            phonetics = features[-1]
            # Some katakana words (e.g., パロメーター) have an "*" instead
            if phonetics == '*':
                phonetics = raw
            if not all((is_katakana(_) for _ in phonetics)):
                raise Exception(f"Not only katakanas in phonetic transcription: {phonetics}")

            ruby = parse_node(node)

        julius = tuple(phonetics_to_julius_phonemes(phonetics)) if phonetics else None
        word = Word(raw, ruby, phonetics, julius)
        words.append(word)

        node = node.next

    return words


def generate_julius_transcript_from_words(words: List['Word']) -> str:
    phoneme_tokens: List[str] = []
    for word in words:
        if not word.phonetics:
            continue
        phoneme_tokens.extend(phonetics_to_julius_phonemes(word.phonetics))
    return ' '.join(phoneme_tokens)


@dataclass
class Word:
    raw: str
    ruby: Optional[Text]
    phonetics: Optional[str]
    julius_phonemes: Optional[Tuple[str, ...]] = None

    def to_html(self) -> str:
        if self.ruby:
            parts = []
            for text, furigana in self.ruby:
                if furigana:
                    parts.append(f"<rb>{text}<rb><rt>{furigana}</rt>")
                else:
                    parts.append(text)
            return f"<ruby>{''.join(parts)}</ruby>"
        else:
            return self.raw


def parsing_test():
    """ Simple parsing test on the JSUT sample data """
    with open('data/jsut_basic5000_sample/transcript_utf8.txt') as f:
        for line in f:
            raw = line.split(':')[1]
            s = Sentence(raw)
            print(s.to_html())


if __name__ == '__main__':
    parsing_test()


def phonetics_to_julius_phonemes(phonetics: Optional[str]) -> List[str]:
    if not phonetics:
        return []
    hiragana = jaconv.kata2hira(phonetics)
    julius = jaconv.hiragana2julius(hiragana)
    tokens = [token for token in julius.split(' ') if token]
    return tokens
