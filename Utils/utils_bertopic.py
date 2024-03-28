# Import packages
import pandas as pd
import numpy as np
from numpy import random
import os
import matplotlib
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import time
from tqdm import tqdm
from adjustText import adjust_text
from scipy.stats import ttest_ind, sem
from scipy import stats
import tiktoken
import ast

# Bertopic
import tensorflow as tf
tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})

# OpenAI
import openai

# Gemini
import google.generativeai as genai

# Taiwan LLAMA
from torch import bfloat16
from transformers import BitsAndBytesConfig
import transformers
import accelerate
import bitsandbytes as bnb
from peft import PeftModel

# Claude 2
import anthropic

# Huggingface
import torch
from transformers import pipeline

# Google colab
from google.colab import userdata
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
CLAUDE_API_KEY = userdata.get('CLAUDE_API_KEY')
HUGGINGFACE_API_KEY = userdata.get('HUGGINGFACE_API_KEY')

# API
openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
client_claude = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

# Hyperparameters
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 30

# Token count of manifestos' length
def show_text_length(column: list, save=False):
    """_summary_
    This function shows the distribution of text length
    
    Args:
        column (list): _description_. Should be a column from a pd.DataFrame
        save (bool, optional): _description_. Defaults to False.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(column, bins=30, alpha=0.7, color='grey', edgecolor="black", linewidth=0.5)
    #   plt.title('Distribution of Content Length of Taiwan Manifesto')
    plt.xlabel('Text Length of Manifestos')
    plt.ylabel('Frequency')
    if save:
        plt.savefig(('Graph/Distribution_token_length'))
    else:
        plt.show()

# Token count
def batch_token_count(df: pd.DataFrame, encoding_name: str, batch_size: int) -> list:
    """
    Count token number per batch from a assigned column within a pd.DataFrame.

    Args:
    - df (pd.DataFrame): Input DataFrame
    - encoding_name (str): Encoding model name (cl100k_base for GPT-4 and all embedding models)
    - batch_size (int): Batch size

    Returns:
    - List[int]: Token number of per batch
    """

    total_tokens = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]['SENTENCE'].tolist()
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = sum([len(encoding.encode(string)) for doc in batch for string in doc])
        total_tokens.append(num_tokens)

    return total_tokens

def generate_target_audience_twllama(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate target audience using Taiwan LLAMA.

    Args:
    - df (pd.DataFrame): Should include topic label, keywords and representative documents

    Returns:
    - pd.DataFrame: 加入目標受眾的 DataFrame。
    """
    
    generation_config = {
            "candidate_count": 1,
            "max_output_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.95,
            }

    # Set up quantization configuration to load large model with less GPU memory
    # This requires the `bitsandbytes` library
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit quantization
        bnb_4bit_quant_type='nf4',  # Normalized float 4
        bnb_4bit_use_double_quant=True,  # Second quantization after the first
        bnb_4bit_compute_dtype=bfloat16  # Computation type
    )
    model_id = "yentinglin/Taiwan-LLM-7B-v2.1-chat"

    # Llama 2 Tokenizer
    tokenizer_tw_llama = transformers.AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_API_KEY)

    # Llama 2 Model
    model_tw_llama = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            ),
        token=HUGGINGFACE_API_KEY,
        do_sample=True,
        top_p=0.95
    )

    peft_path = "dean22029/taiwan_llama_finetuning"
    # model = PeftModel.from_pretrained(model, peft_path)
    model_tw_llama.eval()

    # Our text generator
    generator = transformers.pipeline(
        model=model_tw_llama,
        tokenizer=tokenizer_tw_llama,
        task='text-generation',
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_new_tokens=500,
        repetition_penalty=1.1,
    )

    target_audiences = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating target audiences"):
        documents = ', '.join(row['Representative_Docs'])
        keywords = ', '.join(row['Representation'])
        topic = row['GPT']

        prompt_for_taiwan_llama = f"""
        在以下資訊中，我要求你透過主題的資訊，包含主題標籤、關鍵字與主題下的代表性文件來判別這個主題的潛在目標受眾是誰。
        每個主題都代表著一種類型的選舉政見，請判斷每個主題所訴求的議題或是建設的受益者會是誰，如果這個主題下的政見只會受益到某個特定地區、城市、區域的居民或是民眾，則這個主題的目標受眾就只會有「地區居民」，並回覆為PORK。
        如果你判斷這個主題的政見會使得所有台灣人民，或是不特定範圍的人民受惠，則回覆為POLICY。
        現在，讓我提供一些例子以及標籤劃分的邏輯。
        ---
        主題:台中城市發展及交通建設計畫。
        主題下的代表性政見:
        - 發展 台中 捷運 加速。
        - 台中 機場。
        - 爭取 台中縣 合併 升格 改制 直轄市 加速 台中 都會區 捷運 系統 興建 推動 台中港 兩岸 直航 港口 繁榮 台中 地區 促進 中部 地區 全方位 發展。

        關鍵字: 台中, 台中港, 台中市, 機場, 臺中, 捷運, 中部, 升格, 直航, 大台中
        政見分類（你應該回傳的標籤）: PORK
        標籤邏輯: 在此主題中，主題標籤、關鍵字與代表性文件均顯示出政見只針對「台中」地區的建設，因此不會讓台灣其他地區的人享受到該政見帶來的好處，而是只有台中當地人才會得益於這份政見，綜合上述，這個主題的目標受眾即為「地區居民」，因此回答PORK。
        ---
        主題: 核能發電與再生能源轉型計畫。
        主題下的代表性政見:
        - 能源 政策。
        - 實施 綠色 電價 制度 人民 選舉 綠色 能源 權力 電力 解嚴 打破 台電 能源 壟斷 地位 修改 電業法 推動 電力 自由化。
        - 為了 促進 臺灣 能源 加速 轉型 早日 非核 脫油 國際 能源 價格 飆漲 不至於 吃掉 未來 經濟 成長 成果 因此 主張 核一 三廠 提前 除役 核四 絕不 商轉 再生 能源 ２０１９年 發電量 總發電量 ５％ 成立 能源 世代 轉換 監督 委員會 國安 層級 總統 督導 ９９.４％ 能源 進口 臺灣 進行 能源 結構 轉變 提高 自主 能源 比例 臺灣 ２０２５年 再生 能源 發電 總發電 比例 ２０％。

        關鍵字: 能源, 非核家園, 核四, 核能, 核電, 發電, 核四廠, 廢核, 再生, 除役
        政見分類（你應該回傳的標籤）: POLICY
        標籤邏輯: 這個主題的政見多提到能源政策，在關鍵字和主題標籤均提到核能和再生能源，由於此主題沒有特別提到台灣某處的地名，也沒有特別針對的受眾族群，因此全國人民和不特定多數民眾都可以享受到這份政見帶來的益處，因此本主題的目標受眾應該為「全民」，因此回答POLICY。
        ---
        主題: 運動公園及體育中心發展促進休閒運動與共融親子活動，提供豐富遊具設施，以推動運動發展。
        主題下的代表性政見:
        - 十五 開發 大型 運動 公園 設立 綜合 運動 中心。
        - 爭取 國民 運動 中心。
        - 休閒 運動 公園。

        關鍵字: 運動, 公園, 巨蛋, 體育, 中心, 設施, 休閒, 遊具, 共融, 親子
        政見分類（你應該回傳的標籤）: PORK
        標籤邏輯: 這個主題雖然並沒有具名特別的地區居民，但若判斷主題內的文件和關鍵字，可以看出此主題的政見要爭取建設體育中心或是活動中心，並與親子有關。根據以上資訊，我們可以知道如果建設在某地區的體育中心只能有利於該地區的居民，無法有利於不特定多數的國民，因為非本地區的國民需要額外花費成本來使用這個運動中心，綜合上述，我們將此主題的針對受眾歸類於地區居民，並且因為有利於父母家長親子以及兒童，所以目標受眾應被標注為地區居民，因此回答PORK。

        總結來說，識別目標受眾最關鍵的區別在於判斷該政策是否能夠涵蓋全國範圍或是不特定多數民眾。
        因此，如果主題或關鍵字反覆提到台灣特定地區的居民，他們應該被優先視為「地區居民」，所以你需要回傳PORK。

        我有一個包含以下文件的主題:
        {documents}
        該主題由以下關鍵字描述: {keywords}
        GPT給出的主題標籤是: {topic}

        根據上述的資訊，請你提供這個主題內文件嘗試針對的潛在受眾群體，請你回答是PORK或是POLICY即可，不要加上說明。
        """

        def remove_duplicates(strings_list):
            return list(set(strings_list))

        messages = [
        {
            "role": "system",
            "content": "你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答，並按照用戶要求的格式回答。",
        },
        {"role": "user", "content": prompt_for_taiwan_llama},
        ]

        response = generator(messages)
        target_audiences.append(response[0]["generated_text"][2]['content'])

    df['Target_Audience_Taiwan_LLAMA'] = target_audiences

    return df

def create_prompt(df: pd.DataFrame) -> pd.DataFrame:
    
    for idx, row in df.iterrows():
        documents = ', '.join(row['Representative_Docs'])
        keywords = ', '.join(row['Representation'])
        topic = row['GPT']

        prompt_for_taiwan_llama = f"""
        在以下資訊中，我要求你透過主題的資訊，包含主題標籤、關鍵字與主題下的代表性文件來判別這個主題的潛在目標受眾是誰。
        每個主題都代表著一種類型的選舉政見，請判斷每個主題所訴求的議題或是建設的受益者會是誰，如果這個主題下的政見只會受益到某個特定地區、城市、區域的居民或是民眾，則這個主題的目標受眾就只會有「地區居民」，並回覆為PORK。
        如果你判斷這個主題的政見會使得所有台灣人民，或是不特定範圍的人民受惠，則回覆為POLICY。
        現在，讓我提供一些例子以及標籤劃分的邏輯。
        ---
        主題:台中城市發展及交通建設計畫。
        主題下的代表性政見:
        - 發展 台中 捷運 加速。
        - 台中 機場。
        - 爭取 台中縣 合併 升格 改制 直轄市 加速 台中 都會區 捷運 系統 興建 推動 台中港 兩岸 直航 港口 繁榮 台中 地區 促進 中部 地區 全方位 發展。

        關鍵字: 台中, 台中港, 台中市, 機場, 臺中, 捷運, 中部, 升格, 直航, 大台中
        政見分類（你應該回傳的標籤）: PORK
        標籤邏輯: 在此主題中，主題標籤、關鍵字與代表性文件均顯示出政見只針對「台中」地區的建設，因此不會讓台灣其他地區的人享受到該政見帶來的好處，而是只有台中當地人才會得益於這份政見，綜合上述，這個主題的目標受眾即為「地區居民」，因此回答PORK。
        ---
        主題: 核能發電與再生能源轉型計畫。
        主題下的代表性政見:
        - 能源 政策。
        - 實施 綠色 電價 制度 人民 選舉 綠色 能源 權力 電力 解嚴 打破 台電 能源 壟斷 地位 修改 電業法 推動 電力 自由化。
        - 為了 促進 臺灣 能源 加速 轉型 早日 非核 脫油 國際 能源 價格 飆漲 不至於 吃掉 未來 經濟 成長 成果 因此 主張 核一 三廠 提前 除役 核四 絕不 商轉 再生 能源 ２０１９年 發電量 總發電量 ５％ 成立 能源 世代 轉換 監督 委員會 國安 層級 總統 督導 ９９.４％ 能源 進口 臺灣 進行 能源 結構 轉變 提高 自主 能源 比例 臺灣 ２０２５年 再生 能源 發電 總發電 比例 ２０％。

        關鍵字: 能源, 非核家園, 核四, 核能, 核電, 發電, 核四廠, 廢核, 再生, 除役
        政見分類（你應該回傳的標籤）: POLICY
        標籤邏輯: 這個主題的政見多提到能源政策，在關鍵字和主題標籤均提到核能和再生能源，由於此主題沒有特別提到台灣某處的地名，也沒有特別針對的受眾族群，因此全國人民和不特定多數民眾都可以享受到這份政見帶來的益處，因此本主題的目標受眾應該為「全民」，因此回答POLICY。
        ---
        主題: 運動公園及體育中心發展促進休閒運動與共融親子活動，提供豐富遊具設施，以推動運動發展。
        主題下的代表性政見:
        - 十五 開發 大型 運動 公園 設立 綜合 運動 中心。
        - 爭取 國民 運動 中心。
        - 休閒 運動 公園。

        關鍵字: 運動, 公園, 巨蛋, 體育, 中心, 設施, 休閒, 遊具, 共融, 親子
        政見分類（你應該回傳的標籤）: PORK
        標籤邏輯: 這個主題雖然並沒有具名特別的地區居民，但若判斷主題內的文件和關鍵字，可以看出此主題的政見要爭取建設體育中心或是活動中心，並與親子有關。根據以上資訊，我們可以知道如果建設在某地區的體育中心只能有利於該地區的居民，無法有利於不特定多數的國民，因為非本地區的國民需要額外花費成本來使用這個運動中心，綜合上述，我們將此主題的針對受眾歸類於地區居民，並且因為有利於父母家長親子以及兒童，所以目標受眾應被標注為地區居民，因此回答PORK。

        總結來說，識別目標受眾最關鍵的區別在於判斷該政策是否能夠涵蓋全國範圍或是不特定多數民眾。
        因此，如果主題或關鍵字反覆提到台灣特定地區的居民，他們應該被優先視為「地區居民」，所以你需要回傳PORK。

        我有一個包含以下文件的主題:
        {documents}
        該主題由以下關鍵字描述: {keywords}
        GPT給出的主題標籤是: {topic}
        """
    return prompt_for_taiwan_llama

def generate_target_audience(df: pd.DataFrame, model_name:str) -> pd.DataFrame:
    
    """
    Add columns based on the generation of target audience from LLMs.

    Args:
    - df (pd.DataFrame): Should include topic labels, keywords, and representative documents。
    - model_name: Model name 。

    Returns:
    - pd.DataFrame: With target audience column。
    """
    if model_name not in {'GPT', 'Gemini', 'Taiwan-llama', 'Claude'}:
        print(f"The {model_name} model is not included!")
        return None
    
    generation_config = {
            "candidate_count": 1,
            "max_output_tokens": 256,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            }

    # Gemini Safety setting
    safety_settings = [
        {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
        },
        {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
        },
        {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
        },
        {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
        },
    ]

    if model_name == 'Taiwan-llama':
        # set quantization configuration to load large model with less GPU memory
        # this requires the `bitsandbytes` library
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,  # 4-bit quantization
            bnb_4bit_quant_type='nf4',  # Normalized float 4
            bnb_4bit_use_double_quant=True,  # Second quantization after the first
            bnb_4bit_compute_dtype=bfloat16  # Computation type
        )
        model_id = "yentinglin/Taiwan-LLM-7B-v2.1-chat"

        # Llama 2 Tokenizer
        tokenizer_tw_llama = transformers.AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_API_KEY)

        # Llama 2 Model
        model_tw_llama = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                ),
            token=HUGGINGFACE_API_KEY,
            do_sample=True,
            top_p=0.9
        )

        peft_path = "dean22029/taiwan_llama_finetuning"
        model = PeftModel.from_pretrained(model, peft_path)
        model_tw_llama.eval()


        # Our text generator
        generator = transformers.pipeline(
            model=model_tw_llama,
            tokenizer=tokenizer_tw_llama,
            task='text-generation',
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            max_new_tokens=500,
            repetition_penalty=1.1,
        )

    target_audiences = []

    def remove_duplicates(strings_list):
            return list(set(strings_list))

    # iterate each row
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating target audiences"):
        documents = ', '.join(row['Representative_Docs'])
        keywords = ', '.join(row['Representation'])
        topic = row['GPT']

        # Prompt engineering
        prompt = f"""
        This is a list of texts where each collection of texts describe a topic. After each collection of texts, the name of the topic they represent is mentioned as a short-highly-descriptive title.
        Based on the information provided, you are requested to identify the potential target audience for each topic, considering the topic label, keywords, and representative documents.
        Each topic represents a type of election policy proposal, and you should determine who the beneficiaries of the proposed policies or developments would be.
        If the proposals under a topic would only benefit residents of a SPECIFIC area, city, or region, then the target audience for that topic would be 地區居民.
        If there is a more specific group targeted, it should be noted in parentheses, such as 地區居民 (學生), indicating a focus on students within a particular area, besides if you believe that there are multiple groups within an area will benefit from the policy, use the comma in chinese as 、instead of ,to separate them.
        If you judge that the topic's policy proposals would benefit 'ALL PEOPLE in Taiwan' or 'PEOPLE ACROSS UNSPECIFIED RANGES', then classify according to the given group_list into the relevant groups.
        Also, A topic can have multiple target groups ranging from 1 to 5. Precise labels will be awarded and tipped, make sure the labeling follows the rules above. Now, let me provide you some examples and the logic of labeling.
        The possible groups of target audience are as below:
        audience_list = [全民, 軍公教, 台商, 老人, 婦女, 原住民, 族群, 外籍人士,
                 學生, 中壯年, 青少年, 兒童, 榮民, 勞工, 工商企業,
                 公益團體(社福團體), 專業人士, 特殊技能人士, 弱勢(含性工作者、更生人、卡奴、腳踏車騎士),
                 僑民, 殘障(身心障礙), 失業, 低收入戶, 中間選民, 投資者, 父母家長親子, 單親家庭,
                 選任公務人員, 農漁民, 網民, 地區居民]
        DO NOT include groups or audience that are not in the audience_list!
        If you believe that there's no matching group in the audience_list, assigned the target audience group as 自我宣傳, be awared that there should only be itself if 自我宣傳 is included.
        ---
        Topic: 台中城市發展及交通建設計畫。
        Sample texts from this topic:
        - 發展 台中 捷運 加速。
        - 台中 機場。
        - 爭取 台中縣 合併 升格 改制 直轄市 加速 台中 都會區 捷運 系統 興建 推動 台中港 兩岸 直航 港口 繁榮 台中 地區 促進 中部 地區 全方位 發展。

        Keywords: 台中, 台中港, 台中市, 機場, 臺中, 捷運, 中部, 升格, 直航, 大台中
        Target audience: [地區居民]
        Explanation: Let's think step by step. In this topic, the topic label, keywords, and representative documents all indicate that the policy proposals are specifically aimed at development in the 「台中地區」. Therefore, people from other regions of Taiwan will not enjoy the benefits brought about by these proposals; only the local residents of Taichung will benefit from this policy. In summary, the target audience for this topic is identified as [地區居民].
        ---
        Topic: 核能發電與再生能源轉型計畫。
        Sample texts from this topic:
        - 能源 政策。
        - 實施 綠色 電價 制度 人民 選舉 綠色 能源 權力 電力 解嚴 打破 台電 能源 壟斷 地位 修改 電業法 推動 電力 自由化。
        - 為了 促進 臺灣 能源 加速 轉型 早日 非核 脫油 國際 能源 價格 飆漲 不至於 吃掉 未來 經濟 成長 成果 因此 主張 核一 三廠 提前 除役 核四 絕不 商轉 再生 能源 ２０１９年 發電量 總發電量 ５％ 成立 能源 世代 轉換 監督 委員會 國安 層級 總統 督導 ９９.４％ 能源 進口 臺灣 進行 能源 結構 轉變 提高 自主 能源 比例 臺灣 ２０２５年 再生 能源 發電 總發電 比例 ２０％。

        Keywords: 能源, 非核家園, 核四, 核能, 核電, 發電, 核四廠, 廢核, 再生, 除役
        Target audience: [全民]
        Explanation: Let's think step by step. This topic frequently mentions energy policy, with both keywords and the topic label referring to nuclear energy and renewable resources. Since this topic does not specifically mention any place in Taiwan nor targets any specific group of people, all citizens and a broad, unspecified majority can enjoy the benefits brought by these policy proposals. Therefore, the target audience for this topic should be [全民].
        ---
        Topic: 爭取建設多元親子運動公園，共融設施提供休閒中心。
        Sample texts from this topic:
        - 爭取 經費 增設 全民 運動 中心 打造 特色 公園
        - 爭取 國民 運動 中心。
        - 休閒 運動 公園。

        Keywords: 運動, 公園, 中心, 親子, 遊具, 共融, 休閒, 設施, 爭取, 體育
        In this example, We provide two options for target audience:
        Target audience option1: [地區居民(父母家長親子、兒童)]
        Target audience option2: [地區居民(父母家長親子), 兒童]
        Explanation: Let's think step by step. Although this topic does not specify any particular 地區居民, judging from the documents and keywords within the theme, it can be seen that the policy proposals aim to advocate for the construction of sports centers or activity centers, which are related to parent-child activities. Based on the information provided, we can understand that if a sports center is built in a certain area, it can only benefit the residents of that area and not a broad, unspecified majority of the population.
        This is because nationals from other areas would need to incur additional costs to use this sports center. As a result, option1 is better than option2 since the latter includes 兒童 which is seen as a national wide group. In summary, we categorize the target audience for this topic as local residents, specifically benefiting 父母家長親子, thus it should be labeled as [地區居民(父母家長親子、兒童)], and don't forget the chinese comma '、'.
        ---
        Topic: 以衛武營、鳳山為中心，探討高雄地區文化藝術發展與推動，強調博物館、公園、藝文產業及南方文化等元素，致力於打造高雄成為融合藝術、文化與經濟的城市。
        Sample texts from this topic:
        - 衛武營 流行 音樂 中心 時代 來臨 強化 在地 文化 產業 扶植 高雄 藝文 專業 人才。
        - 文化 高雄 藝文 城市 打造 高雄 成為 流行 文化 舉辦 藝術 文化節 提倡 藝文 表演 推動 演唱會 經濟 藝文 產業 深耕 扶植 動漫 電競 產業 鼓山 鹽埕 金新興 苓雅 成為 藝術 亮點 串聯 衛武營 文化 中心 中央 公園 海音 中心 駁二 哈瑪星 美術館 內惟 藝術 中心 凹子 森林 公園 創造 藝術 文化 經濟 活化 公有 閒置 場域 推展 親子 婦幼 長輩 青年 活動 空間 發展 高雄 經濟。
        - 打造 文化 鳳山 中央 協助 電影 造景 模式 再現 黃埔 新村 風華 打造 唯一 眷村 文化 博物館 推動 五甲 國宅 變身 吸引 藝術家 進駐 成為 衛武營 兩廳院 商業 人才 支援 系統 力爭 政府 扶植 南方 藝術 表演 人才 大東 園區 形成 藝術 黃金 三角 聚落 鳳山 文化 偉大。

        Keywords: 衛武營, 鳳山, 高雄, 黃埔, 公園, 藝術, 藝文, 博物館, 文化, 南方
        Target audience: [地區居民(工商企業、 公益團體(文化團體)、 自我宣傳、 青年、 專業人士)]
        Explanation: Let's think step by step. Based on the sample text, keywords, and topic, we can understand that the focus of this theme is on the art and culture industry in Kaohsiung City, along with related professionals, public welfare groups, and youth, rather than the national art and culture industry and related individuals nationwide. Therefore, the target audience for this theme should be 地區居民(工商企業, 公益團體(文化團體), 自我宣傳, 青年, 專業人士), categorizing all groups under local residents.
        ---
        Topic:嘉義都市建設與繁榮發展。
        Sample texts from this topic:
        - 促成 嘉義市 中央 部會 嘉義 地區 交通 改善 繁榮 遠景 共同 協力 打造 嘉義 成為 雲嘉南 重點 交通 樞紐 轉運 中心
        - 全力 促使 嘉義 師範 學院 嘉義 技術 學院 合併 成為 嘉義 大學
        - 爭取 嘉義 技術 學院 嘉義 師範 學院 升格為 嘉義 大學

        keywords: ['嘉義', '嘉義市', '嘉義縣', '高鐵', '大嘉義', '爭取', '建設', '繁榮', '國立', '觀光']
        In this example, We provide two options for target audience:
        Target audience option1:[地區居民(學生)]
        Target audience option2:[地區居民, 學生]
        Explanation: Let's think step by step. Based on the above information, option1 is better than option2 because this topic specifically mentioned 嘉義 or 嘉義市 or 嘉義 大學 which are restricted to certain area instead of national wide.

        In summary, the most crucial distinction in identifying the target audience lies in determining whether the policy can encompass a national scope.
        Therefore, if the topic or keywords repeatedly mention residents of specific areas in Taiwan, they should be prioritized as 地區居民 and should not include 全民.
        If other groups are also included at the same time, they should be noted in parentheses, for example: 地區居民(學生).
        Remember, a precise labeling will be awarded and tipped. Again, make sure you follow the instructions I mentioned.

        I have a topic that contains the following documents:
        {documents}
        The topic is described by the following keywords: {keywords}
        The topic label by made by GPT is: {topic}

        Based on the information above, provide a list of the potential audience that the documents within the topic trying to target and return the following format, need no to provide the explanation:
        audience: [group1, group2, ...]
        """

        if model_name == 'GPT':
            response = client_openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=150,
                top_p=TOP_P,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            audience_response = response.choices[0].message.content.strip()
            target_audience = audience_response.split(":")[1].strip().strip("[]").replace('"', '').split(", ")
            target_audiences.append(remove_duplicates(target_audience))
            time.sleep(4) # to prevent RateLimitErrors

        elif model_name == 'Gemini':
            model = genai.GenerativeModel('gemini-pro',
                generation_config=generation_config,
                safety_settings=safety_settings,
                )
            response = model.generate_content(prompt)
            audience_response = response.text.strip()
            target_audience = audience_response.split(":")[1].strip().strip("[]").replace('"', '').split(", ")
            target_audiences.append(remove_duplicates(target_audience))
            time.sleep(4) # to prevent RateLimitErrors

        elif model_name == 'Taiwan-llama':
            messages = [
            {
                "role": "system",
                "content": "你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答，並按照用戶要求的格式回答。",
            },
            {"role": "user", "content": prompt},
            ]
            response = generator(messages)
            target_audiences.append(response[0]["generated_text"][2]['content'])

        elif model_name == 'Claude':
            message = client_claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                )
            audience_response = message.content[0].text
            print(audience_response)
            target_audience = audience_response.split("audience:")[1].strip().strip("[]").replace('"', '').split(", ")
            target_audiences.append(remove_duplicates(target_audience))
            time.sleep(4) # to prevent RateLimitErrors

        else:
            raise ValueError(f"{model_name} is not a valid model name.")

    if model_name == 'GPT':
        df['Target_Audience_GPT'] = target_audiences
    elif model_name == 'Gemini':
        df['Target_Audience_Gemini'] = target_audiences
    elif model_name == 'Claude':
        df['Target_Audience_Claude'] = target_audiences
    elif model_name == 'Taiwan-llama':
        df['Target_Audience_Taiwan_LLAMA'] = target_audiences

    return df

# Function of checking model outputs
def str_to_list(string: str or list) -> list:
    if type(string) == str:
        try:
            return ast.literal_eval(string)
        except ValueError:
            return string.strip("[]").replace('\'', '').split(", ")
    elif type(string) == list:
        return string

# Function of similarity check from different models
def similarity_check(df: pd.DataFrame) -> int:
    """_summary_
    Similarity check of different LLMs.
    
    Args:
        df (pd.DataFrame): _description_. Should be a topic info DataFrame.

    Returns:
        int
    """
    gpt_gemini = df['PORK_GPT'] == df['PORK_Gemini']
    gpt_claude = df['PORK_GPT'] == df['PORK_Claude']
    gemini_claude = df['PORK_Gemini'] == df['PORK_Claude']
    similarity = (df['PORK_GPT'] == df['PORK_Gemini']) & (df['PORK_GPT'] == df['PORK_Claude']) & (df['PORK_Gemini'] == df['PORK_Claude'])
    print(f"Similarity of GPT vs. Gemini: {round(gpt_gemini.sum() / len(df), 3)}")
    print(f"Similarity of GPT vs. Claude: {round(gpt_claude.sum() / len(df), 3)}")
    print(f"Similarity of Claude vs. Gemini: {round(gemini_claude.sum() / len(df), 3)}")
    print(f"Similarity of three models: {round(similarity.sum()  / len(df), 3)}")

    return gpt_gemini.sum(), gpt_claude.sum(), gemini_claude.sum(), similarity.sum()

def assign_pork(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign PORK value based on Target_Audience_GPT & Target_Audience_Claude.

    Args:
    - df (pd.DataFrame): Should include Target_Audience_GPT, Target_Audience_Gemini, and Target_Audience_Claude.

    Returns:
    - pd.DataFrame: With PORK_AI and PORK columns.
    """

    df['PORK_AI'] = pd.np.where(
        df['PORK_GPT'] == df['PORK_Claude'],
        df['PORK_Claude'], # True
        -1 # False: require human check
    )

    return df


def pork_version(df: pd.DataFrame, topic_v: str) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame)
        topic_v (str): Topic_info_df version

    Returns:
        pd.DataFrame
    """
    topic_info = pd.read_csv(f'Result/Result_{topic_v}/topic_info_{topic_v}.csv')
    pork = topic_info[topic_info['PORK'] == 1]['Topic'].tolist()
    df[f'PORK_{topic_v}'] = df['Topic'].apply(lambda x: 1 if x in pork else 0)
    return df

def merge_and_classify(df: pd.DataFrame, version_names: list) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame)
        version_names (list)

    Returns:
        pd.DataFrame
    """
    result_df = pd.DataFrame()
    result_df['ID'] = df['ID'].unique()

    for version_name in version_names:
        df[f'WEIGHT_{version_name}'] = df['WEIGHT'] * df[f'PORK_{version_name}']
        df[f'PART_{version_name}'] = df['PART'] * df[f'PORK_{version_name}']

        grouped_weight = df.groupby('ID')[f'WEIGHT_{version_name}'].sum().reset_index(name=f'WEIGHT_PORK_{version_name}')
        grouped_part = df.groupby('ID')[f'PART_{version_name}'].sum().reset_index(name=f'PART_PORK_{version_name}')

        result_df = result_df.merge(grouped_weight, on='ID', how='left')
        result_df = result_df.merge(grouped_part, on='ID', how='left')

    return result_df

def plot_pork_policy_ratios(dfs: list, titles: list, save=False):
    """
    Plot a single bar chart with stacked bars showing the ratio of pork to policy topics in each version.

    Args:
    - dfs (list): List of DataFrames, each containing the data for one version.
    - titles (list): List of version titles (strings) to be used as bar labels.
    """

    width = 0.3
    ratios = []
    for df in dfs:
        pork_ratio = df['PORK'].mean() # 假設'PORK'列包含的是1和0，其中1表示 pork
        print(f"PORK: {df['PORK'].sum()}, TOTAL: {df.shape[0]}")
        policy_ratio = 1 - pork_ratio
        ratios.append((pork_ratio, policy_ratio))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#6E7F80', '#A4BE8C']

    # plot pork ratio of different version
    for i, (pork, policy) in enumerate(ratios):
        ax.bar(titles[i], pork, color=colors[0], width=width, label='Pork' if i == 0 else "")
        ax.bar(titles[i], policy, bottom=pork, color=colors[1], width=width, label='Policy' if i == 0 else "")

    ax.set_ylabel('Ratio')
    # ax.set_title('Pork vs Policy Topics Ratio by Version')
    ax.legend(loc='best')

    plt.tight_layout()
    if save:
        plt.savefig('Graph/Pork_vs_Policy_Topics_Ratio_by_Version.png')
    else:
        plt.show()

def plot_change(df: pd.DataFrame, columns: list, save=False):
    """_summary_

    Args:
        df (pd.DataFrame)
        columns (list)
        save (bool, optional): _description_. Defaults to False.
    """
    colors = ['#6E7F80']
    fig, ax = plt.subplots(figsize=(8, 6))

    width_ = 0.3
    means = []
    errors = []

    for th in sorted(df['TH'].unique()):
        df_filtered = df[df['TH'] == th]
        mean = df_filtered[columns].mean().mean()
        std = df_filtered[columns].std().mean()  # standard error
        print(f"mean: {mean}, std: {std}")
        n = len(df_filtered)
        se = std / np.sqrt(n)
        confidence_interval = se * stats.t.ppf((1 + 0.95) / 2., n-1)  # 95% CI

        means.append(mean)
        errors.append(confidence_interval)

    positions = np.arange(len(means))
    ax.bar(positions, means, width=width_, color=colors[0], yerr=errors, capsize=5, error_kw={'elinewidth': 2, 'ecolor': 'black'})

    ax.set_xlabel('Election')
    ax.set_ylabel('Average Pork Ratio')
    plt.xticks(positions, ['6th Election', '7th Election'])
    plt.legend(loc='best', labels=['Mean of Pork Ratio'])
    plt.tight_layout()
    
    if save:
        plt.savefig('Graph/Average_Weights_by_Election_TH.png')
    else:
        plt.show()


def perform_t_test(df: pd.DataFrame, columns: list, candidate: str):
    """
    Conduct T test of 6th and 7th election and plot the results including confidence intervals.

    Args:
    df -- DataFrame containing the data.
    columns -- List of columns representing different pork ratios.
    candidate -- Name of the candidate or group being tested.
    """
    mean_values_th1 = df[df['TH'] == 6][columns].mean(axis=1)
    mean_values_th2 = df[df['TH'] == 7][columns].mean(axis=1)

    # Perform t-test
    t_stat, p_value = ttest_ind(mean_values_th1, mean_values_th2, equal_var=False, alternative='less')

    # Displaying results
    print(f"T-statistic: {t_stat}, P-value: {p_value}")
    if p_value < 0.05:
        print(f"{candidate}間存在顯著差異。")
    else:
        print(f"{candidate}間不存在顯著差異。")

    return t_stat, p_value

def plot_change_each(df: pd.DataFrame, columns: list, save=False):
    colors = ['#6E7F80', '#D9AF6B', '#A4BE8C', '#D1B2A5', '#957DAD']
    fig, ax = plt.subplots(figsize=(8, 6))

    width = 0.1
    averages = []
    for index, column in enumerate(columns):
        df_filtered = df[df['TH'].isin([6, 7])]
        grouped = df_filtered.groupby('TH')[column].mean()
        th_averages = df_filtered.groupby('TH')[columns].mean().mean()
        averages.append(th_averages)
        positions = np.arange(len(grouped)) + (width * index)
        ax.bar(positions, grouped, width=width, label=column, color=colors[index % len(colors)])

    # ax.set_title('Average Ratio by Election TH')
    ax.set_xlabel('Election')
    ax.set_ylabel('Average Pork Ratio')
    th_year_dict = {6: "2004", 7: "2008"}
    new_xticks = [f"{int(th)}\n({th_year_dict.get(th, '')})" for th in grouped.index]
    plt.xticks(np.arange(len(grouped)) + width * (len(columns) - 1) / 2, grouped.index)
    ax.set_xticklabels(new_xticks)
    legend_labels = [f'Version {i}' for i in range(1, len(columns) + 1)]
    plt.legend(loc='upper center', labels=legend_labels)
    plt.tight_layout()
    if save:
        plt.savefig('Graph/Average_Weights_by_Election_TH.png')
    else:
        plt.show()
        
def visualization(df: pd.DataFrame, model: str, party: str, y_axis: int, columns: list, reform_year=2005, save=False, avg=False):
    """
    Visualization function, make sure the dataframe contains specified columns in `columns` list, e.g., 'TH', 'PORK_v1', 'PORK_v2', and 'POLICY'.

    Args:
    - df (pd.DataFrame): DataFrame containing the final results.
    - model (str): Name of the model.
    - party (str): Name of the party.
    - y_axis (int): The position of the line plot.
    - columns (list): List of column names to be visualized.
    - reform_year (int): The year of electoral reform.
    - save (bool): Whether to save the plot as a file.
    """
    print(f'Number of candidates: {len(df)}')
    country = 'Taiwan' if reform_year == 2005 else 'Japan'

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#6E7F80', '#D9AF6B', '#A4BE8C', '#D1B2A5', '#957DAD']  # More colors can be added if needed
    averages = []

    if avg:
        df['avg_all_columns'] = df[columns].mean(axis=1)
        overall_avg = df.groupby('TH')['avg_all_columns'].mean()
        overall_avg.plot(ax=ax, label='Average Pork Ratio', color=colors[0], marker='o', linestyle='-', linewidth=2)
    else:
        for index, column in enumerate(columns):
            grouped = df.groupby('TH')[column].mean()
            policy_ratios = 1 - grouped
            grouped.plot(ax=ax, label=column, color=colors[index % len(colors)], marker='o', linestyle='-')

            result_df = pd.DataFrame({
            'TH': grouped.index,
            'PORK': grouped.values,
            'POLICY': policy_ratios.values
            })

        # texts = []
        # for i, j in zip(grouped.index, grouped.values):
        #     texts.append(ax.text(i, j, f'{j:.2f}', ha='center', va='bottom', color=colors[index % len(colors)]))

    # Title, labels, and lines
    ax.set_title(f'Evolution of Pork Ratios in {party} Candidates Across Elections')
    ax.set_xlabel('Election Year')
    ax.axvline(x=6.25, color='black', linestyle='-')
    ax.annotate('Electoral Reform', xy=(6.25, y_axis), xytext=(6.35, y_axis + 0.01), arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    # X-tick labels
    th_year_dict = {2: "1992", 3: "1995", 4: "1998", 5: "2001", 6: "2004", 7: "2008", 8: "2012", 9: "2016", 10: "2020", 11: "2024"}
    ax.set_xticks(np.arange(2, len(th_year_dict) + 2))  # Corrected to set xticks based on 'TH' values
    # ax.set_xticklabels([th_year_dict.get(th) for th in range(2, len(th_year_dict) + 2)], fontsize=10)  # Updated for dynamic x-tick labels
    new_xtick_labels = [f"{th}\n({year})" for th, year in th_year_dict.items()]
    ax.set_xticklabels(new_xtick_labels, fontsize=10)
    ax.set_ylabel('Average Pork Ratio', fontsize=12)


    if avg:
        ax.legend(loc='best', fontsize=10)
    else:
        lines, labels = ax.get_legend_handles_labels()
        legend_labels = [f'Version {i}' for i in range(1, len(columns) + 1)]
        ax.legend(lines, legend_labels, loc='best')

    if save:
        if avg:
            plt.savefig(f'Graph/{country}_Pork_{model}_{party}_avg.png')
        else:
            plt.savefig(f'Graph/{country}_Pork_{model}_{party}.png')
    else:
        plt.show()

def translation(df: pd.DataFrame, columns: list):
    for col in columns:
        translate_text = []
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating translation"):
            text = row[col]
            prompt = f"Translate the following text to English: {text}, and please be concise, you need not to provide the original text."
            message = client_claude.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]}
                ],
                )
            trans_text = message.content[0].text.strip()
            translate_text.append(trans_text)
            time.sleep(3)
        df[col] = translate_text
    return df