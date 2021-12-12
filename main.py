import argparse
from collections import defaultdict

import MeCab
import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from sklearn.feature_extraction.text import TfidfTransformer
from wordcloud import WordCloud
import matplotlib.pyplot as plt


def main(keyword):
    urls = search(keyword, num_results=30)
    df = pd.DataFrame([page_words(url) for url in urls], index=urls).fillna(0)
    tfidf = TfidfTransformer(smooth_idf=False)
    transformed_df = pd.DataFrame(tfidf.fit_transform(df).toarray(), columns=df.columns, index=urls)
    wc = WordCloud(background_color='white', font_path="~/Library/Fonts/ipaexg.ttf").generate_from_frequencies(
        transformed_df.sum().to_dict())
    fig = plt.figure(figsize=(12, 12))
    plt.suptitle(f"Keyword: {keyword}", fontsize=20)
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.imshow(wc)
    ax1.axis("off")
    ax2 = fig.add_subplot(2, 1, 2)
    transformed_df.sum().sort_values(ascending=False).head(20).plot.bar(ax=ax2)
    plt.savefig(f"{keyword}.png")


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
    parser = argparse.ArgumentParser(description="US Kabu Bot")
    parser.add_argument("--keyword", help="Set search word.")
    args = parser.parse_args()
    main(args.keyword)
