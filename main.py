from collections import defaultdict

import MeCab
import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search


def main(keyword):
    urls = search(keyword, num_results=30)
    df = pd.DataFrame([page_words(url) for url in urls]).fillna(0)
    print(df.shape)


def word_count(text):
    word_freq = defaultdict(int)
    mecab = MeCab.Tagger("-d /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd")
    for line in mecab.parse(text).removesuffix("\nEOS\n").split("\n"):
        word, info = line.split("\t")
        if info.split(",")[0] == "名詞":
            word_freq[word] += 1
    return word_freq


def page_words(url):
    res = requests.get(url)
    bs = BeautifulSoup(res.content, "html.parser")
    return word_count(bs.text)


if __name__ == "__main__":
    main("投資")
