import numpy as np
import pandas as pd
import re
import time
from pathlib import Path
from tqdm.auto import tqdm
from ckiptagger import WS, POS, NER, construct_dictionary

# CKIP module — model data must be downloaded separately to CKIP_PATH
CKIP_PATH = str(Path(__file__).resolve().parents[2] / "CKIP_TAGGER")
ws = WS(CKIP_PATH)
pos = POS(CKIP_PATH)
ner = NER(CKIP_PATH)

# Self-define dict
word_to_weight = {'823': 1, 'ECFA': 1, '2300': 1, '台26線': 1, '台74線': 1, '12年國教': 1, 'BOT': 1, '88快速道路': 1, '台27線': 1, '台61線': 1, '十二年國教': 1, '國道10號': 1,
                  '台88號': 1, 'M型': 1, '205兵工廠': 1, '北二高': 1, '台65線': 1, 'CEPA': 1, 'FTA': 1, '科學園區': 1, '228': 1, 'MIT': 1, '202兵工廠': 1, '86快速道路': 1, '國道8號': 1,
                  '台64': 1, '台66': 1, 'iBike': 1, 'MRT': 1, 'TPP': 1, 'TIFA': 1, 'TPP':1, '台22': 1, '台29': 1, '國10': 1, '國1': 1, '318': 1, 'NCC':1, 'PM2.5': 1, 'YouBike': 1,
                  '台68': 1, '快速道路': 1, 'NGO': 1, 'NPO': 1, 'U-Bike': 1, 'LGBTQ': 1, '三七五減租': 1, '小三通': 1, '大三通': 1, '基礎建設': 1, '戒急用忍': 1, '社會役': 1, '非核家園': 1,
                  '教育券': 1, '九二共識': 1}
dictionary = construct_dictionary(word_to_weight)

# Stopwords — resolved relative to project root
_STOPWORDS_PATH = Path(__file__).resolve().parents[2] / "data" / "stopwords_zh-tw.txt"
with open(_STOPWORDS_PATH, encoding="utf-8") as fin:
    stopwords = fin.read().split("\n")[1:]


def flatten(input_list: list) -> list:
  return sum(input_list, [])

# Text filter
def text_select(ws_sentence: list, pos_sentence: list, bert=False) -> list:
    """
    Text selection based on listed conditions.

    Args:
        ws_sentence (list): word segmentation results.
        pos_sentence (list): pos results.
        bert (bool, optional): Defaults to False.

    Returns:
        list:
    """
    assert len(ws_sentence) == len(pos_sentence)

    sentence_list = []
    clean_pos = []

    stop_pos = set(['Nep', 'Nh', 'Neqa', 'Neu'])
    allowed_pos = ('V', 'N', 'A', 'D')

    re_email = re.compile(r'\S+@\S+')
    re_url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    re_phone_number = re.compile(r'\d{3}-\d{4}-\d{4}|\d{10}')

    for word, pos in zip(ws_sentence, pos_sentence):
        if bert:
            valid_word = len(word) > 1 and not re_email.match(word) and not re_url.match(word) and not re_phone_number.match(word)
        else:
            valid_word = len(word) > 1 and word not in stopwords and not re_email.match(word) and not re_url.match(word) and not re_phone_number.match(word) and pos.startswith(allowed_pos) and pos not in stop_pos

        if valid_word:
            sentence_list.append(word)
            clean_pos.append(f"{word}({pos})")

    sentence = " ".join(sentence_list)

    if bert:
        return sentence
    else:
        return sentence, clean_pos


def load_text(content: str, test=False) -> str:
    ws_ = ws([content], recommend_dictionary=dictionary)
    pos_ = pos(ws_)

    (token, clean_pos) = text_select(flatten(ws_), flatten(pos_))
    sentence = text_select(flatten(ws_), flatten(pos_), bert=True)

    if test:
        ner_ = ner(ws_, pos_)
        return sentence, clean_pos, ner_, token
    else:
         return sentence, token

def full_to_half(text):
        n = []
        for char in text:
            code = ord(char)
            if code == 12288:
                code = 32
            elif 65281 <= code <= 65374:
                code -= 65248
            n.append(chr(code))
        return ''.join(n)

def remove_list_marks(text: str) -> str:
    """
    移除句首的標號，但保留其他所有的文字。
    """
    patterns = [
        r'^\d+[,．，、。\.]',
        r'^[一二三四五六七八九十零]+[,，、。\.]',
        r'^\([一二三四五六七八九十零\d]+\)',
        r'^\（[一二三四五六七八九十零\d]+\）',
        r'^\〔[一二三四五六七八九十零\d]+\〕',
        r'^[①②③④⑤⑥⑦⑧⑨⑩]+[,，、。\.]',
        r'^[１２３４５６７８９０]+[,，、。\.]',
        r'^[ABCDEFGHIJK]+[,，、。\.]'
        r'^[壹貳參肆伍陸柒捌玖拾]+[,，、。\.]'
        r'^[●■⊙※*@.#◎▶★>▓▲◆©]',
        r'[㈠-㈩]+[,，、。\.]',
        r'[⑴-⒇]+[,，、。\.]',
    ]

    text = full_to_half(text)
    for pattern in patterns:
        text = re.sub(pattern, '', text)

    return text.strip()

def tokenization(year: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        year (int): Election year.
        df (pd.DataFrame): Dataframe that has CONTENT column

    Returns:
        pd.DataFrame: A tokenized dataframe.
    """
    start = time.time()

    df['CONTENT'] = df['CONTENT'].apply(remove_list_marks)
    df['CONTENT_LENGTH'] = df['CONTENT'].apply(len)
    tqdm.pandas(desc=f"Tokenizing {year} election statements")
    df[['SENTENCE', 'TOKEN']] = df['CONTENT'].progress_apply(load_text).apply(pd.Series)

    end = time.time()
    print(f"{year}年選舉公報的斷詞運算時間為: {round((end - start) / 60, 2)} 分")
    return df

def split_content(df: pd.DataFrame) -> pd.DataFrame:
    """
    Divide manifestos into individual policy sentences.
    """
    rows_list = []

    for _, row in df.iterrows():
        content = row['CONTENT']
        newline_count = content.count('\n')

        if newline_count > 3:
            sentences = re.split(r'\n|(\n[。！？])', content)
        else:
            sentences = re.split(r'[。！]', content)

        sentences = [s for s in sentences if s and not s.isspace()]

        for sentence in sentences:
            sentence = sentence.strip()
            sentence = sentence.replace(' ', '').replace('\n', '')

            if sentence.endswith((':', '：')):
                continue
            new_row = row.copy()
            new_row['CONTENT'] = sentence
            rows_list.append(new_row)

    new_df = pd.concat([pd.DataFrame([row]) for row in rows_list], ignore_index=True)

    return new_df

def postprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Postprocessing: remove newlines and filter short CONTENT entries.
    """

    def remove_newlines(text):
        if pd.isnull(text):
            return text
        else:
            text = text.strip()
        return text.replace('\n', '').replace('\r', '')

    filtered_df = df[df['CONTENT'].str.len() >= 2].copy()
    filtered_df['CLEAN_CONTENT'] = filtered_df['CONTENT'].apply(remove_list_marks)
    filtered_df['WEIGHT'] = filtered_df['CONTENT'].apply(lambda x: len(x)) / filtered_df.groupby('ID')['CONTENT'].transform('sum').apply(lambda x: len(x))
    filtered_df['PART'] = 1 / filtered_df.groupby('ID')['ID'].transform('size')

    for column in ['CONTENT', 'SENTENCE', 'TOKEN', 'CLEAN_CONTENT']:
        if column in filtered_df.columns:
            filtered_df.loc[:, column] = filtered_df[column].apply(remove_newlines)
            filtered_df[column] = filtered_df[column].astype(str)

    filtered_df = filtered_df.dropna()

    return filtered_df
