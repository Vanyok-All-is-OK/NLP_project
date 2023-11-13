import re
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

np.random.seed(100)


MIN_WORD_SIZE = 1
regex = re.compile("[А-Яа-яA-z]{" + str(MIN_WORD_SIZE) + ",}")
lemmatizer = WordNetLemmatizer()

pat1 = r'@[A-Za-z0-9]+'
links_reg = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, links_reg))


def train_val_test_split(df, train_size, val_size, test_size, stratify):
    train, val = train_test_split(df, train_size=train_size, stratify=df[stratify])
    val, test = train_test_split(val, train_size=val_size / (val_size + test_size), stratify=val[stratify])
    return train, val, test


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts.
    Source: https://webdevblog.ru/podhody-lemmatizacii-s-primerami-v-python/
    """
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)


def words_only(text, regex=regex):
    return regex.findall(text.lower())


@lru_cache(maxsize=128)
def lemmatize_word(word):
    return lemmatizer.lemmatize(word, get_wordnet_pos(word))


def lemmatize_text(text):
    return [
        lemmatize_word(word) 
        for word in text 
        if word not in stopwords.words('english')
    ]


def clean_text(text):
    tokens = words_only(text)
    lemmas = lemmatize_text(tokens)
    return ' '.join(lemmas)


def lemmatize_texts(texts, use_cache=True):
    with Pool(4) as p:
        lemmas = list(tqdm(
            p.imap(clean_text, texts),
            total=len(texts)
        ))
    return lemmas

