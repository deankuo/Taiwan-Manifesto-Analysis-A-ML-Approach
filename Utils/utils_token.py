import numpy as np
import pandas as pd
import re
import os
import time
from tqdm.auto import tqdm
from ckiptagger import WS, POS, NER, construct_dictionary # tokenization

# CKIP module
CKIP_PATH = "../CKIP_TAGGER"
ws = WS(CKIP_PATH) # 斷詞
pos = POS(CKIP_PATH) # 詞性標註
ner = NER(CKIP_PATH) # 命名實體識別

# 加入自定義字典
word_to_weight = {'823': 1, 'ECFA': 1, '2300': 1, '台26線': 1, '台74線': 1, '12年國教': 1, 'BOT': 1, '88快速道路': 1, '台27線': 1, '台61線': 1, '十二年國教': 1, '國道10號': 1,
                  '台88號': 1, 'M型': 1, '205兵工廠': 1, '北二高': 1, '台65線': 1, 'CEPA': 1, 'FTA': 1, '科學園區': 1, '228': 1, 'MIT': 1, '202兵工廠': 1, '86快速道路': 1, '國道8號': 1,
                  '台64': 1, '台66': 1, 'iBike': 1, 'MRT': 1, 'TPP': 1, 'TIFA': 1, 'TPP':1, '台22': 1, '台29': 1, '國10': 1, '國1': 1, '318': 1, 'NCC':1, 'PM2.5': 1, 'YouBike': 1, 
                  '台68': 1, '快速道路': 1, 'NGO': 1, 'NPO': 1, 'U-Bike': 1, 'LGBTQ': 1, '三七五減租': 1, '小三通': 1, '大三通': 1, '基礎建設': 1, '戒急用忍': 1, '社會役': 1, '非核家園': 1,
                  '教育券': 1, '九二共識': 1}
dictionary = construct_dictionary(word_to_weight)

# 停用詞
with open("./Data/stopwords_zh-tw.txt", encoding="utf-8") as fin:
    stopwords = fin.read().split("\n")[1:]
    
def flatten(input_list: list) -> list:
  return sum(input_list, [])

# Text filter
def text_select(ws_sentence: list, pos_sentence: list, bert=False) -> list:
    assert len(ws_sentence) == len(pos_sentence)
    
    sentence_list = []
    clean_pos = []
    
    stop_pos = set(['Nep', 'Nh', 'Neqa', 'Neu']) # 名詞中的代名詞、數量詞、詞綴都不保留（From Amy's book）
    allowed_pos = ('V', 'N', 'A', 'D')  # 保留動詞、名詞、形容詞與副詞
    
    # Detect email, url, and phone number
    re_email = re.compile(r'\S+@\S+')
    re_url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    re_phone_number = re.compile(r'\d{3}-\d{4}-\d{4}|\d{10}')

    for word, pos in zip(ws_sentence, pos_sentence):
        # 考慮 bert=True 盡可能保留完整的詞意（如果使用embedding應該可以整個句子丟下去就好）
        if bert:
            # 保留停用詞 & 一般詞性
            valid_word = len(word) > 1 and not re_email.match(word) and not re_url.match(word) and not re_phone_number.match(word)
        else:
            valid_word = len(word) > 1 and word not in stopwords and not re_email.match(word) and not re_url.match(word) and not re_phone_number.match(word) and pos.startswith(allowed_pos) and pos not in stop_pos
        
        if valid_word:
            sentence_list.append(word) 
            clean_pos.append(f"{word}({pos})") 

    sentence = " ".join(sentence_list) # 串成一個字串

    if bert:
        return sentence
    else:
        return sentence, clean_pos


def load_text(content: str, test=False) -> str:
    ws_ = ws([content], recommend_dictionary=dictionary) # double lists
    pos_ = pos(ws_) # double lists
    
    (token, clean_pos) = text_select(flatten(ws_), flatten(pos_)) # token for LDA
    sentence = text_select(flatten(ws_), flatten(pos_), bert=True) # sentence for BERTopic
    
    # Test 
    if test:
        ner_ = ner(ws_, pos_) # dictionaries in list
        return sentence, clean_pos, ner_, token
    else:
         return sentence, token
     
def full_to_half(text):
        n = []
        for char in text:
            code = ord(char)
            if code == 12288:  # 全形空格直接转换
                code = 32
            elif 65281 <= code <= 65374:  # 全形字符（除空格）转换公式
                code -= 65248
            n.append(chr(code))
        return ''.join(n)

def remove_list_marks(text):
    """移除句首的標號，但保留其他所有的文字。"""
    patterns = [
        r'^\d+[．，、\.]',  # 數字加點，例如 "1."
        r'^[一二三四五六七八九十零]+[，、\.]',  # 中文數字，例如 "一、"，"十一、"
        r'^\([一二三四五六七八九十零\d]+\)',  # 括號的中文數字，例如 "(1)"，"(十一)"
        r'^\（[一二三四五六七八九十零\d]+\）',
        r'^[①②③④⑤⑥⑦⑧⑨⑩]',
        r'^[１２３４５６７８９０]+[．\.]',
        r'^●',
        r'^■',
        r'^⊙',
        r'^\*',
        r'^@',
        r'^\.',
        r'^#',
        r'^◎',
        r'^▶',
        r'^★',
        r'[㈠-㈩]',
        r'[\u2474-\u2487]', # 類似這種：⑵
    ]
    
    text = full_to_half(text)
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    
    return text.strip()

def tokenization(year: int, df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

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
    
    # 每一年份使用CPU的運算時間約為10分鐘
    print(f"{year}年選舉公報的斷詞運算時間為: {round((end - start) / 60, 2)} 分")
    return df

def split_content(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    This function is used to divide the manifestos into policies or sentences.
    Args:
        df (pd.DataFrame): Input a Dataframe that contains CONTENT column which stores the manifesto data.

    Returns:
        pd.DataFrame: A dataframe that contains the divided manifestos so it should be larger.
    """
    rows_list = []
    
    for _, row in df.iterrows():
        content = row['CONTENT']
        newline_count = content.count('\n')
        
        # 以換行符號決定切分依據
        if newline_count > 5:
            # 使用換行符號和句號、驚嘆號切分
            sentences = re.split(r'\n|(\n[。！])', content)
        else:
            # 使用句號切分
            sentences = re.split(r'[。！]', content)
        
        # 移除切分结果中的空字串，包含最後一句
        sentences = [s for s in sentences if s and not s.isspace()]
        
        # 複製每一筆原本的資料，對齊切分後的筆數
        for sentence in sentences:
            sentence = sentence.strip()
            sentence = sentence.replace(' ', '').replace('\n', '')
            # 如果是以冒號結尾的句子則刪除
            if sentence.endswith((':', '：')):
                continue
            new_row = row.copy()
            new_row['CONTENT'] = sentence
            rows_list.append(new_row)

    new_df = pd.concat([pd.DataFrame([row]) for row in rows_list], ignore_index=True)
    
    return new_df

def postprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Postprocessing DataFrame: 去除換行符號以及 CONTENT 字串中長度小於 2 的資料。

    Args:
        df (pd.DataFrame): 已經切分好的 DataFrame。

    Returns:
        pd.DataFrame: 後處理的 DataFrame，包含 CLEAN_CONTENT, WEIGHT, PART 欄位。
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
