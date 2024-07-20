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
from typing import Tuple
import ast

# Bertopic
import tensorflow as tf
tf.config.optimizer.set_jit(True)
tf.config.optimizer.set_experimental_options({'auto_mixed_precision': True})

# sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from pprint import pprint

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
from google.colab import drive
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
    """
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

    return sum(np.array(total_tokens) > 8191)

def plot_grid_search_result(model: GridSearchCV, save=True):
    """
    Given a trained GridSearchCV object, plot the mean test scores for each learning rate and topic number combination.

    Args:
        model (GridSearchCV): GridSearchCV object
        save (bool, optional): _description_. Defaults to True.
    """
    # Extract results
    results = model.cv_results_
    learning_rates = sorted(list(set(params['learning_decay'] for params in results['params'])))
    n_components_list = sorted(list(set(params['n_components'] for params in results['params'])))

    plt.figure(figsize=(12, 6))

    for rate in learning_rates:
        scores_for_rate = []

        for n_components in n_components_list:
            index = next(i for i, params in enumerate(results['params'])
                         if params['learning_decay'] == rate and params['n_components'] == n_components)
            scores_for_rate.append(results['mean_test_score'][index])

        plt.plot(n_components_list, scores_for_rate, marker='o', label=f'Learning Rate: {rate}')

    plt.xticks(n_components_list)
    plt.xlabel('Number of Topics')
    plt.ylabel('Mean Test Score')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig('Graph/Grid Search Results_Large.png')
    else:
        plt.show()

# Create Document — Topic Matrix
def document_topic_matrix(df_: pd.DataFrame, lda_model: LatentDirichletAllocation, data: list) -> Tuple[pd.DataFrame, list]:
    """
    Constructs a document-topic matrix based on the output from an LDA model.

    Parameters:
    - df_: pd.DataFrame, a DataFrame containing the original documents, must include an 'ID' column.
    - lda_model: LatentDirichletAllocation, the trained LDA model.
    - data: list, the data of documents to be analyzed, typically a list of preprocessed text data.

    Returns:
    - df_document_topic: pd.DataFrame, a table with the distribution of topics per document, including the weight of each topic and the most dominant topic.
    - topicnames: list, a list of topic names, based on the number of components in the model.

    Description:
    - Firstly, the function transforms the provided data using the LDA model to get the distribution of each document across the topics.
    - A DataFrame is created containing the topic distributions for each document.
    - The dominant topic for each document is calculated.
    - Document IDs and their dominant topics are added to the DataFrame.
    """
    lda_output = lda_model.transform(data)

    # column names
    topicnames = [f"Topic {i}" for i in range(lda_model.n_components)]

    # Make the pandas dataframe
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames)

    # Get dominant topic for each document
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['Dominant_topic'] = dominant_topic
    df_document_topic['ID'] = df_['ID']

    return df_document_topic, topicnames

def get_combined_topic_info(vectorizer: CountVectorizer, lda_model: LatentDirichletAllocation, df_document_topic: pd.DataFrame, df: pd.DataFrame, n_keywords=10, n_docs=3):
    """
    Extracts and combines key information for each topic from an LDA model, including
    keywords and the most representative documents for each topic.

    Parameters:
    - vectorizer: The vectorizer used to encode the documents. This must have a method
      `get_feature_names_out()` to get the list of token names.
    - lda_model: The trained LatentDirichletAllocation model from which to extract topic
      information.
    - df_document_topic: DataFrame containing the topic distribution per document.
    - df: The original DataFrame with documents and their corresponding IDs.
    - n_keywords: int, number of top keywords to extract for each topic (default is 10).
    - n_docs: int, number of top documents to be identified as most representative for each
      topic (default is 3).

    Returns:
    - DataFrame containing each topic with its keywords and representative documents.

    Description:
    - Step 1: Extracts the top `n_keywords` for each topic from the LDA model components
      using the feature names obtained from the `vectorizer`.
    - Step 2: Identifies the top `n_docs` for each topic based on the highest topic
      contributions in `df_document_topic`. It uses the document IDs to link back to the
      original documents.
    - Step 3: Creates a final DataFrame combining the topics, their keywords, and
      a list of tokens from the most representative documents for each topic.
    """
    # Step 1: Extract keywords for each topic
    keywords = np.array(vectorizer.get_feature_names_out())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_keywords]
        topic_keywords.append(keywords.take(top_keyword_locs))

    # Step 2: Identify the top n documents for each topic
    df_copy = df_document_topic.copy()
    df_dropped = df_document_topic.drop(columns=['ID', 'Dominant_topic'])
    top_doc = []
    for topic in df_dropped.columns:
        top_docs_for_topic = df_dropped[topic].nlargest(n_docs).index
        top_docs = df_copy.loc[top_docs_for_topic, 'ID'].tolist()
        for doc in top_docs:
            top_doc.append({'Topic': topic, 'ID': doc})

    df_top_docs = pd.DataFrame(top_doc)
    df_top_docs = pd.merge(df_top_docs, df[['ID', 'TOKEN']], how='inner', on='ID')
    df_top_docs = df_top_docs.groupby('Topic')['TOKEN'].agg(list).reset_index()

    assert len(topic_keywords) == len(df_top_docs)

    # Step 3: Create the final DataFrame
    df_final = pd.DataFrame({
        'Topic': range(len(topic_keywords)),
        'Keywords': topic_keywords
    })
    df_final = pd.merge(df_final, df_top_docs)
    df_final = df_final.rename(columns={'TOKEN': 'Representative_Docs'})

    return df_final

# Create Topic Distribution
def topic_distribution(df_document_topic: pd.DataFrame) -> pd.DataFrame:
    df =  df_document_topic['Dominant_topic'].value_counts().reset_index(name="Document Count")
    df.columns = ['Topic', 'Document Count']
    return df

def show_topic_keyword(vectorizer: CountVectorizer, lda_model: LatentDirichletAllocation, n_keywords: int = 10) -> pd.DataFrame:
    """
    Displays the top `n_keywords` for each topic in a Latent Dirichlet Allocation (LDA) model.

    Parameters:
    - vectorizer (CountVectorizer): The vectorizer that was used to transform the text data into a matrix.
      It must have a method `get_feature_names_out()` to retrieve the list of all features.
    - lda_model (LatentDirichletAllocation): The trained Latent Dirichlet Allocation model from which the topic keywords
      will be extracted.
    - n_keywords (int): The number of top keywords to display for each topic (default is 10).

    Returns:
    - pd.DataFrame: A DataFrame where each column represents a topic, and each row lists a keyword from
      that topic in order of importance.

    Description:
    - The function extracts the feature names from the `vectorizer`.
    - For each topic in the LDA model, it identifies the `n_keywords` with the highest
      weights.
    - A DataFrame is created where each column corresponds to a topic and the rows
      contain the top keywords for that topic.
    - The index of the DataFrame is set to indicate the rank of each keyword within the
      topic.
    """
    keywords = np.array(vectorizer.get_feature_names_out())
    topic_keywords = []

    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_keywords]
        topic_keywords.append(keywords.take(top_keyword_locs))

    
    df = pd.DataFrame(topic_keywords).transpose()
    df.index = ['Word ' + str(i) for i in range(df.shape[0])]
    df.columns = ['Topic ' + str(i) for i in range(df.shape[1])]

    return df

def get_model_response(client, model_name: str, prompt: str, config: dict) -> str:
    """
    Sends a prompt to a specified AI model and returns the model's response.

    Args:
    - client: The API client configured for the model.
    - model_name (str): Name of the model (e.g., 'GPT', 'Claude').
    - prompt (str): The input prompt for the model.
    - config (dict): Configuration parameters for the API call such as temperature, top_p, and top_k.

    Returns:
    - str: The response text from the AI model.
    """
    if model_name == 'GPT':
        response = client.Completion.create(
            engine="gpt-4o",
            prompt=prompt,
            temperature=config['temperature'],
            max_tokens=config['max_output_tokens'],
            top_p=config['top_p'],
            stop=None
        )
        return response['choices'][0]['text'].strip()

    elif model_name == 'Claude':
        response = client.Message.create(
            model="claude-3-sonnet-20240229",
            prompt=prompt,
            max_tokens=config['max_output_tokens'],
            temperature=config['temperature'],
            top_p=config['top_p'],
            stop=None
        )
        return response['choices'][0]['message']['content'].strip()

    else:
        raise ValueError(f"{model_name} is not a valid model name.")

def generate_target_audience(df: pd.DataFrame, model_name: str, config: dict, unit: str = 'manifesto', BERTopic: bool = False) -> pd.DataFrame:
    """
    Generates target audience profiles based on provided DataFrame using multiple generative AI models.

    Args:
    - df (pd.DataFrame): DataFrame containing documents, keywords, and topic tags.
    - model_name (str): Name of the model to be used for generation. Valid models are 'GPT', 'Gemini', 'Taiwan-llama', 'Claude'.
    - config (dict): Dictionary of generation configuration setting.
    - unit (str): The unit of analysis, default as manifesto.
    - BERTopic (Bool): Whether the model is BERTopic or not, default as False.

    Returns:
    - pd.DataFrame: Updated DataFrame with the target audiences added.
    
    Raises:
    - ValueError: If the provided model name is not supported.
    """
    if model_name not in {'GPT', 'Gemini', 'Taiwan-llama', 'Claude'}:
        raise ValueError(f"The {model_name} model is not included!")

    
    # Initialize API clients based on model
    if model_name == 'Claude':
        client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    elif model_name == 'GPT':
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
    else:
        raise ValueError(f"{model_name} is not a valid model name.")

    target_audiences = []
    explanations = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating target audiences"):
        documents = "- " + "\n\t- ".join(row['Representative_Docs'])
        if BERTopic:
            keywords = ', '.join(row['Representation'])
            topic = row['GPT']
        else:
            keywords = ', '.join(row['Keywords'])

        # Constructing prompt for AI model
        if unit == 'manifesto':
            prompt = f"""
            This is a list of topics each represented by a collection of texts and keywords. You are requested to analyze the information and categorize each topic into potential target audiences based on the following guidelines:

            - Firstly, consider the keywords.
            - Secondly, consider the texts within the topic.
            - Finally, comprehensively consider the keywords and texts within the topic.
            - Each topic relates to a specific type of policy proposal.
            - Identify who the beneficiaries of these policies would be.
            - If the benefits are restricted to a specific area, label the target audience as 地區居民. Include specific sub-groups in parentheses if applicable (e.g., [地區居民(學生)]).
            - Use the semicolon (;) to separate multiple groups within the same area (e.g., [地區居民(學生; 工商企業)]) and use the comma to separate different groups in the audience_list (e.g., [地區居民(學生; 工商企業), 自由行旅客]).
            - If the policy benefits all people in Taiwan or across unspecified regions, classify them into broader groups provided in the list below.
            - A topic can target 1 to 5 groups. Accurate and precise labels are crucial.
            - In summary, the most crucial distinction in identifying the target audience lies in determining whether the policy can encompass a national scope.

            List of possible target audiences:
            audience_list = [全民, 軍公教, 台商, 老人, 婦女, 原住民, 族群(閩南、客家族群、眷村), 外籍人士, 學生, 中壯年, 青年, 兒童, 榮民, 勞工, 藝文人士, 工商企業, 醫療人員, 病人, 選手, 公益團體(社福團體), 專業人士, 社工員, 自由行旅客, 特殊技能人士, 弱勢(含性工作者、更生人、卡奴、腳踏車騎士), 僑民, 殘障(身心障礙), 失業, 中低收入戶, 中間選民, 投資者, 父母家長親子, 單親家庭, 選任公務人員(議員、里長], 農漁民, 網民, 地區居民]
            DO NOT include groups not in the audience_list. If no appropriate group fits, label the target audience as [自我宣傳], and be aware that it should only be itself if 自我宣傳 is included.
            Here are some examples:

            -----
            The topic keywords: 推動, 爭取, 新竹, 中央, 補助, 提高, 都市, 市民, 更新, 安全
            Explanation: The topic keywords indicate that this topic aimed at subsidies of safety in Hsinchu City which only benefit the residents nearby. Therefore, the target audience for this topic is [地區居民].
            Target audience: [地區居民]
            -----
            The topic keywords: 推動, 爭取, 新竹, 中央, 補助, 提高, 都市, 市民, 更新, 安全
            Sample texts from this topic:
            - 相挺 專業 認真 立委 守護 環境 永續 發展 堅持 不懈 修正 完成 礦業法 重要 改革 獲得 公民 監督 國會 聯盟 會期 特殊 貢獻獎 揭發 塑膠 地墊 塑化劑 超標 要求 標檢局 全面 塑膠 地墊 市購 檢驗 守護 孩子 健康 環境 永續 。 捍衛 勞動 工作 安全 健康 完成 勞工 職業 災害 保護法 整合 預防 補償 重建 擴大 職災 勞工 保護 成功 爭取 警察 制服 換季 自主權 避免 熱傷 害損 健康 完成 高樓大廈 冷氣 安裝 維修 指導 原則 設計 規劃 施工 維修 空間 行車 安全 里程碑 建立 臺灣 車輛 撞擊 測試 制度 提供 消費者 公開 完整 車輛 安全 資訊 消費者 安全 車輛 選擇 降低 道路 意外 事故 傷亡率 。 強化 銀新 未來城 公益性 要求 擴大 日照 中心 服務 量能 納入 市價 租金 友善 銀髮宅 規劃 蘆洲 五股 三重 交通 便利 成功 爭取 補助 環狀 捷運 北環段 帶動 蘆洲 三重 東區 都市 發展 中央 補助 永康 公園 長安 立體 停車場 興建 蘆 洲 停車 空間 更多 中央 覈定 五泰 輕軌 可行性 評估 五股 交通 便利 安全 成功 爭取 電塔 下地 溪墘 變電所 室內化 計畫 強化 電韌性 協調 蘆洲 五股 興珍 市區 高壓 電塔 地下化 爭取 行政院 覈定 變電所 室內化 改建 計畫 陸續 完成 市區 電桿 地下化 完成 五股 民義路 電桿 下地 三重 碧華 國小 電桿 下地 持續 施作 蘆洲 復興路 民族路 電桿 下地 蘆洲 永樂街 永平街 32巷 電桿 下地 。 督促 地方 建設 施工 中央 全額 負擔 工程款 交流道 改善 工程 動工 中央 覈定 補助 汙水 下水道 倒閉 工程 決標 復工 爭取 中央 全額 負擔 五股 堤防 增高 工程 守護 五股 居民 生命 財產 安全 爭取 中央 建設 五股 環境 治理 親水 空間 營造 109年 投資 五股 地區 改善 觀音山 硬漢嶺 通訊 品質 成功 協調 中華 電信 完成 基地臺 架設 緊急 救援 阻礙 階段 持續 電信 業者 溝通 協調 。 爭取 空大 校園 社區化 空中 大學 階段 拆除 保和街 中正路 圍牆 形成 友善 人行 步道 空間 升級 翻新 學校 建設 國中 小學 冷氣 設施 設備 跑道 操場 老舊 廁所 全面 更新 爭取 經費 超過 學生 開心 家長 放心,
            - 潘懷宗 安心 承諾 嚴打 黑心 把關 食安 強化 食品 藥品 源頭 管制 審核 機制 食藥署 規範 食品 藥品 建立 完整 資料庫 民眾 查詢 建置 食品 建立 重罰 重懲 機制 查出 黑心 食品 至少 以上 上架 違規 情節 嚴重 商品 永遠 重新 上架 二 因應 老年 社會 落實 長照 研擬 社區 照護 中心 增加 國人 參與 長照 服務 誘因 長照法 人手 不足 偏遠 地區 設施 缺乏 財源 短缺 問題 未來 逐漸 檢討 長照法 提供 足夠 誘因 鼓勵 國人 參與 廣設 社區 照護 中心 整合 長照 服務網 就近 提供 長者 失能 使用 三 提升 社會 福利 關心 弱勢 研擬 降低 中低收入戶 申請 門檻 中低收入戶 標準 家庭 平均 所得 生活費 低收入戶 標準 家庭 平均 所得 生活費 生活 消費 支出 提升 跟著 調整 生活費 標準 幫助 弱勢 族羣 四 鼓勵 青年 安心 成家 檢討 現行 貸款 限制 研擬 提高 低利 貸款 額度 內政部 青年 成家 方案 期程 屆滿 續辦 影響 目前 僅存 財政部 公股 銀行 配合 青年 安心 成家 購屋 優惠 貸款 內政部 整合 住宅 補貼 資源 實施 方案 無法 滿足 青年人 購屋 需求 未來 政策性 房貸 提出 檢討 研擬 提高 低利 貸款 額度 五 打造 友善 士林 大同 改善 交通 環境 加速 都市 更新 促進 在地 經濟 蘭州 斯文 整宅 完工 居住 空間 老舊 問題 日趨 嚴重 加強 都市 更新 速度 交通 基礎建設 發展 特色 觀光 士林 著重區 特色 發展 持續 促成 士林 科技 園區 完工 提升 在地 經濟,
            - 林德福 年輕人 試試 1 增加 立體化 停車場 2 提升 公託 育兒 補助 3 加速 雙和 進度
            Explanation: Let's think step by step. First, the topic keywords are aimed at [地區居民].
                            Secondly, the sample texts are specifically aimed at development in 五股, 蘆洲, 三重 and 士林, which are also specific regions in Taiwan, so the sample texts also aimed at [地區居民(勞工; 兒童; 弱勢(含性工作者、更生人、卡奴、腳踏車騎士); 青年; 老人; 中低收入戶; 父母家長親子)].
                            Finally, people from other regions of Taiwan will not benefit from these proposals; only the local residents of 新竹 and 臺北 will. Therefore, the target audience for this topic is [地區居民(勞工; 兒童; 弱勢(含性工作者、更生人、卡奴、腳踏車騎士); 青年; 老人; 中低收入戶; 父母家長親子)].
            Target audience: [地區居民(勞工; 兒童; 弱勢(含性工作者、更生人、卡奴、腳踏車騎士); 青年; 老人; 中低收入戶; 父母家長親子)]
            -----
            The topic keywords: 桃園, 基隆, 捷運, 龍潭, 中壢, 航空城, 推動, 平鎮, 基隆市, 桃園縣
            Explanation: The topic keywords indicate that this topic aimed at MRT construction in Keelung City Taoyuan City which only benefit the residents nearby. Therefore, the target audience for this topic is [地區居民].
            Target audience: [地區居民]
            -----
            The topic keywords: 桃園, 基隆, 捷運, 龍潭, 中壢, 航空城, 推動, 平鎮, 基隆市, 桃園縣
            Sample texts from this topic:
            - 興建 碼頭 強化 基隆港 貨櫃 中心 競爭力 符合 世代 航運 競爭 需求。2 實現 基隆港 港埠 設施 全面 現代化 自動化 電氣化 提升 服務 經營 效率 3 規劃 西岸 碼頭 成為 世代 會議 展覽 觀光 特區 重現 港岸 周邊 地區 歷史 景觀 配合 火車站 計畫 基隆 曼妙 港都 風情 4 協助 基隆市 政府 基隆 港務局 爭取 補修 市區 道路 經費 。 5 打造 跨港 纜車線 海科館 西岸 碼頭 會展 觀光 特區 串連 重要 觀光 景點 型塑 基隆 獨特 山海 觀光 魅力 6 籌劃 臺灣 近代 戰爭 歷史 博物館 結合 大武崙 砲臺 情人湖 風景區 突顯出 基隆 獨特 歷史 地位 打造 深具 文史 內涵 國家級 觀光 特區 7 爭取 基隆市 海岸線 納入 東北角 宜蘭 海岸 國家 風景區 北海岸 觀音山 國家 風景區 基隆 擁有 國家級 風景區 。 8 爭取 臺北 捷運 板南線 南港 延伸至 汐止 基隆 。 9 廢除 中山 汐止 收費站 北二高 七堵 收費站 10 解決 通勤 推動 基隆市 公車 客運 社區 發車 直通 大臺北 都會 11 支持 在地 藝文 團隊 推動 文化 創意 產業 基隆 生根
            - 阿甘 願景 誠信 活力 開放 專業 創新 理念 問政 建立 臺灣人 一致 中心 價值 地方 智慧 融合 國際 視野 立足 臺灣 爭取 中央 政府 預算 資源 建設 國家 大門 桃園 真正 具有 國際 水準 國際 航空城 行動 綱領 設立 甘國秀 地方 民意 委員會 推派 委員 召開 委員會 決議 作成 立法院 問政 方針 追蹤 提報 成效 落實 徵詢 民意 機制 提升 桃園 基礎 雙語 教育 水準 苦學 經驗 建立 雙語 教學 師資 教育 環境 訓練 檢覈 制度 基礎 教育 師資 根基 美哉 桃園 山水 連線 航空 專業 推動 設立 安全 經濟 飛艇 航運 事業 原住民 鄉親 移民 桃園 鄉親 空中 鳥瞰 桃園 山地 飄到 海岸 創新 觀光 價值 伸張 毒瘤 稅留 普世 公理 汙染 工業 合理 回饋 爭取 經費 回饋 桃園 鄉民 落實 地方 建設 推動 飛越 通航 飛越 大陸 領空 到達 節省 成本 提升 航空 產業 競爭力 工作 子弟 阿扁 民航 學校 民國 91年 阿扁 風光 剪綵 作秀 民航 學校 無疾而終 學子 飛行 夢想 墜落 民航 專技 教育 併入 桃園 現有 技術 學院 培養 民航 專業 人才 落實 航空 成立 民航 專業 技師 公會 整合 持照 民航 人員 建立 專業 公信 組織 監督 主管 機關 專業 維護 勞工 權益 強化 飛安 水準 落實 合一 彙整 專業 資源 空港 桃園 空港 管理 納入 地方 充實 地方 資源
            - 配合 民主進步黨 執政 團隊 制訂 相關 法案 爭取 桃園 建設 加速 興建 桃園 航空城 高鐵 聯外 交通網 桃園 捷運 系統 桃園 客貨運 中心 桃園 科學園區 推動 桃園 中壢 鐵路 高架化 打造 桃園 黃金海岸 開闢 桃園 國際 商港 興建 桃園 人工湖 解決 桃園 民生 工業 用水 問題 督促 政府 落實 照顧 弱勢 族羣 強化 社會 福利 推動 本土化 教育 臺灣 臺灣 推動 制訂 憲法 支持 臺灣 正名 運動 臺灣 走向 世界
            Explanation: Let's think step by step. First, the topic title and keywords are aimed at [地區居民].
                            Secondly, the sample texts are specifically aimed at development in 基隆 and 桃園, which are also specific regions in Taiwan, so the sample texts also aimed at [地區居民(弱勢(含性工作者、更生人、卡奴、腳踏車騎士); 中低收入戶; 原住民; 學生; 青年; 藝文人士)].
                            Finally, people from other regions of Taiwan will not benefit from these proposals; only the local residents of 基隆 and 桃園 will. Therefore, the target audience for this topic is [地區居民(工商企業; 弱勢(含性工作者、更生人、卡奴、腳踏車騎士); 中低收入戶; 原住民; 學生; 青年; 藝文人士)].
            Target audience: [地區居民(工商企業; 弱勢(含性工作者、更生人、卡奴、腳踏車騎士); 中低收入戶; 原住民; 學生; 青年; 藝文人士)]
            -----

            The topic keywords: {keywords}
            Sample texts from this topic:
            {documents}

            Based on the information above, provide a list of the potential audience that the documents within the topic trying to target and return the following format, need no to provide the explanation.
            Return format: Target audience: [group1, group2, ...]
            """
        else:
            if BERTopic:
                prompt = f"""
                This is a list of topics each represented by a collection of texts, keywords, and a descriptive title. You are requested to analyze the information and categorize each topic into potential target audiences based on the following guidelines:

                - Firstly, consider the keywords.
                - Secondly, consider the texts within the topic.
                - Finally, comprehensively consider the topic title, keywords, and texts within the topic.
                - Each topic relates to a specific type of policy proposal.
                - Identify who the beneficiaries of these policies would be.
                - If the benefits are restricted to a specific area, label the target audience as 地區居民. Include specific sub-groups in parentheses if applicable (e.g., [地區居民(學生)]).
                - Use the semicolon (;) to separate multiple groups within the same area (e.g., [地區居民(學生; 工商企業)]) and use the comma to separate different groups in the audience_list (e.g., [地區居民(學生; 工商企業), 自由行旅客]).
                - If the policy benefits all people in Taiwan or across unspecified regions, classify them into broader groups provided in the list below.
                - A topic can target 1 to 5 groups. Accurate and precise labels are crucial.
                - In summary, the most crucial distinction in identifying the target audience lies in determining whether the policy can encompass a national scope.

                List of possible target audiences:
                audience_list = [全民, 軍公教, 台商, 老人, 婦女, 原住民, 族群(閩南、客家族群、眷村), 外籍人士, 學生, 中壯年, 青年, 兒童, 榮民, 勞工, 藝文人士, 工商企業, 醫療人員, 病人, 選手, 公益團體(社福團體), 專業人士, 社工員, 自由行旅客, 特殊技能人士, 弱勢(含性工作者、更生人、卡奴、腳踏車騎士), 僑民, 殘障(身心障礙), 失業, 中低收入戶, 中間選民, 投資者, 父母家長親子, 單親家庭, 選任公務人員(議員、里長), 農漁民, 網民, 地區居民]
                DO NOT include groups not in the audience_list. If no appropriate group fits, label the target audience as [自我宣傳], and be aware that it should only be itself if 自我宣傳 is included.
                Here are some examples:

                -----
                Keywords: 台中, 台中港, 台中市, 機場, 臺中, 捷運, 中部, 升格, 直航, 大台中
                Explanation: The topic title and keywords indicate that this topic specifically aimed at infrastructure development in 台中地區. Therefore, the target audience for this topic is [地區居民].
                Target audience: [地區居民]
                -----
                Keywords: 台中, 台中港, 台中市, 機場, 臺中, 捷運, 中部, 升格, 直航, 大台中
                Sample texts from this topic:
                - 發展 台中 捷運 加速
                - 台中 機場
                - 爭取 台中縣 合併 升格 改制 直轄市 加速 台中 都會區 捷運 系統 興建 推動 台中港 兩岸 直航 港口 繁榮 台中 地區 促進 中部 地區 全方位 發展
                Explanation: Let's think step by step. First, the topic title and keywords are aimed at [地區居民]. Secondly, the sample texts are specifically aimed at development in 台中地區. Finally, people from other regions of Taiwan will not benefit from these proposals; only the local residents of Taichung will. Therefore, the target audience for this topic is [地區居民].
                Target audience: [地區居民]
                -----
                Keywords: 能源, 非核家園, 核四, 核能, 核電, 發電, 核四廠, 廢核, 再生, 除役
                Explanation: The topic title and keywords indicate that this topic aimed at nuclear power and renewable energy which benefits all residents in Taiwan. Therefore, the target audience for this topic is [全民].
                Target audience: [全民]
                -----
                Keywords: 能源, 非核家園, 核四, 核能, 核電, 發電, 核四廠, 廢核, 再生, 除役
                Sample texts from this topic:
                - 能源 政策。
                - 實施 綠色 電價 制度 人民 選舉 綠色 能源 權力 電力 解嚴 打破 台電 能源 壟斷 地位 修改 電業法 推動 電力 自由化
                - 為了 促進 臺灣 能源 加速 轉型 早日 非核 脫油 國際 能源 價格 飆漲 不至於 吃掉 未來 經濟 成長 成果 因此 主張 核一 三廠 提前 除役 核四 絕不 商轉 再生 能源 ２０１９年 發電量 總發電量 ５％ 成立 能源 世代 轉換 監督 委員會 國安 層級 總統 督導 ９９.４％ 能源 進口 臺灣 進行 能源 結構 轉變 提高 自主 能源 比例 臺灣 ２０２５年 再生 能源 發電 總發電 比例 ２０％
                Explanation: Let's think step by step. First, the topic title and keywords are aimed at [全民]. Secondly, The sample texts often discuss energy policy and and are not limited to specific regions. Finally, all residents of Taiwan stand to benefit from these proposals. Therefore, the target audience for this topic is [全民].
                Target audience: [全民]
                -----
                Topic title: 爭取建設多元親子運動公園，共融設施提供休閒中心
                Keywords: 運動, 公園, 中心, 親子, 遊具, 共融, 休閒, 設施, 爭取, 體育
                Explanation: The topic title and keywords indicate that this topic aimed at construction of sports centers or activity centers which only benefit the residents nearby. Therefore, the target audience for this topic is [地區居民(父母家長親子; 兒童)].
                Target audience: [地區居民(父母家長親子; 兒童)]
                -----
                Topic title: 爭取建設多元親子運動公園，共融設施提供休閒中心
                Keywords: 運動, 公園, 中心, 親子, 遊具, 共融, 休閒, 設施, 爭取, 體育
                Sample texts from this topic:
                - 爭取 經費 增設 全民 運動 中心 打造 特色 公園
                - 爭取 國民 運動 中心
                - 休閒 運動 公園
                Explanation: Let's think step by step. First, the topic title and keywords are aimed at [地區居民(父母家長親子; 兒童)]. Secondly, the sample texts aim to advocate for the construction of sports centers or activity centers, which are related to parent-child activities. Finally, based on the information provided, constructing a sports center in a particular area will primarily benefit only the local residents of that area rather than a wide, unspecified majority. This is because nationals from other areas would need to incur additional costs to use this sports center.
                Therefore, [地區居民(父母家長親子), 兒童] is incorrect since it includes 兒童 which is seen as a national wide group. In summary, we categorize the target audience for this topic as local residents, specifically benefiting 父母家長親子, thus it should be labeled as [地區居民(父母家長親子; 兒童)].
                Target audience: [地區居民(父母家長親; 兒童)]
                -----
                Topic title: 以衛武營、鳳山為中心，探討高雄地區文化藝術發展與推動，強調博物館、公園、藝文產業及南方文化等元素，致力於打造高雄成為融合藝術、文化與經濟的城市
                Keywords: 衛武營, 鳳山, 高雄, 黃埔, 公園, 藝術, 藝文, 博物館, 文化, 南方
                Explanation: The topic title and keywords indicate that this topic aimed at the art and culture industry in Kaohsiung City. Therefore, the target audience for this theme should be [地區居民(公益團體(文化團體))].
                Target audience: [地區居民(公益團體(文化團體))]
                -----
                Topic title: 以衛武營、鳳山為中心，探討高雄地區文化藝術發展與推動，強調博物館、公園、藝文產業及南方文化等元素，致力於打造高雄成為融合藝術、文化與經濟的城市
                Keywords: 衛武營, 鳳山, 高雄, 黃埔, 公園, 藝術, 藝文, 博物館, 文化, 南方
                Sample texts from this topic:
                - 衛武營 流行 音樂 中心 時代 來臨 強化 在地 文化 產業 扶植 高雄 藝文 專業 人才
                - 文化 高雄 藝文 城市 打造 高雄 成為 流行 文化 舉辦 藝術 文化節 提倡 藝文 表演 推動 演唱會 經濟 藝文 產業 深耕 扶植 動漫 電競 產業 鼓山 鹽埕 金新興 苓雅 成為 藝術 亮點 串聯 衛武營 文化 中心 中央 公園 海音 中心 駁二 哈瑪星 美術館 內惟 藝術 中心 凹子 森林 公園 創造 藝術 文化 經濟 活化 公有 閒置 場域 推展 親子 婦幼 長輩 青年 活動 空間 發展 高雄 經濟
                - 打造 文化 鳳山 中央 協助 電影 造景 模式 再現 黃埔 新村 風華 打造 唯一 眷村 文化 博物館 推動 五甲 國宅 變身 吸引 藝術家 進駐 成為 衛武營 兩廳院 商業 人才 支援 系統 力爭 政府 扶植 南方 藝術 表演 人才 大東 園區 形成 藝術 黃金 三角 聚落 鳳山 文化 偉大
                Explanation: Let's think step by step. First, the topic title and keywords are aimed at  [地區居民(公益團體(文化團體))]. Secondly, the sample texts aim to adcovate for the art and culture industry in Kaohsiung City, along with related professionals, public welfare groups, and youth. Finally, people from other regions of Taiwan will not benefit from these proposals; only the local residents of Kaohsiung will. Therefore, the target audience for this topic is [地區居民(工商企業; 公益團體(文化團體); 青年; 專業人士)].
                Target audience: [地區居民(工商企業; 公益團體(文化團體); 青年; 專業人士)]
                -----

                The topic title: {topic}
                The topic keywords: {keywords}
                Sample texts from this topic:
                {documents}

                Based on the information above, provide a list of the potential audience that the documents within the topic trying to target and return to the following format, need no provided the explanation.
                Return format: Target audience: [group1, group2, ...]
                """
            else:
                prompt = f"""
                This is a list of topics each represented by a collection of texts, keywords, and a descriptive title. You are requested to analyze the information and categorize each topic into potential target audiences based on the following guidelines:

                - Firstly, consider the topic title and keywords.
                - Secondly, consider the texts within the topic.
                - Finally, comprehensively consider the keywords, and texts within the topic.
                - Each topic relates to a specific type of policy proposal.
                - Identify who the beneficiaries of these policies would be.
                - If the benefits are restricted to a specific area, label the target audience as 地區居民. Include specific sub-groups in parentheses if applicable (e.g., [地區居民(學生)]).
                - Use the semicolon (;) to separate multiple groups within the same area (e.g., [地區居民(學生; 工商企業)]) and use the comma to separate different groups in the audience_list (e.g., [地區居民(學生; 工商企業), 自由行旅客]).
                - If the policy benefits all people in Taiwan or across unspecified regions, classify them into broader groups provided in the list below.
                - A topic can target 1 to 5 groups. Accurate and precise labels are crucial.
                - In summary, the most crucial distinction in identifying the target audience lies in determining whether the policy can encompass a national scope.

                List of possible target audiences:
                audience_list = [全民, 軍公教, 台商, 老人, 婦女, 原住民, 族群(閩南、客家族群、眷村), 外籍人士, 學生, 中壯年, 青年, 兒童, 榮民, 勞工, 藝文人士, 工商企業, 醫療人員, 病人, 選手, 公益團體(社福團體), 專業人士, 社工員, 自由行旅客, 特殊技能人士, 弱勢(含性工作者、更生人、卡奴、腳踏車騎士), 僑民, 殘障(身心障礙), 失業, 中低收入戶, 中間選民, 投資者, 父母家長親子, 單親家庭, 選任公務人員(議員、里長), 農漁民, 網民, 地區居民]
                DO NOT include groups not in the audience_list. If no appropriate group fits, label the target audience as [自我宣傳], and be aware that it should only be itself if 自我宣傳 is included.
                Here are some examples:

                -----
                Keywords: 台中, 台中港, 台中市, 機場, 臺中, 捷運, 中部, 升格, 直航, 大台中
                Explanation: The keywords indicate that this topic specifically aimed at infrastructure development in 台中地區. Therefore, the target audience for this topic is [地區居民].
                Target audience: [地區居民]
                -----
                Keywords: 台中, 台中港, 台中市, 機場, 臺中, 捷運, 中部, 升格, 直航, 大台中
                Sample texts from this topic:
                - 發展 台中 捷運 加速
                - 台中 機場
                - 爭取 台中縣 合併 升格 改制 直轄市 加速 台中 都會區 捷運 系統 興建 推動 台中港 兩岸 直航 港口 繁榮 台中 地區 促進 中部 地區 全方位 發展
                Explanation: Let's think step by step. First, the keywords are aimed at [地區居民]. Secondly, the sample texts are specifically aimed at development in 台中地區. Finally, people from other regions of Taiwan will not benefit from these proposals; only the local residents of Taichung will. Therefore, the target audience for this topic is [地區居民].
                Target audience: [地區居民]
                -----
                Keywords: 能源, 非核家園, 核四, 核能, 核電, 發電, 核四廠, 廢核, 再生, 除役
                Explanation: The keywords indicate that this topic aimed at nuclear power and renewable energy which benefits all residents in Taiwan. Therefore, the target audience for this topic is [全民].
                Target audience: [全民]
                -----
                Keywords: 能源, 非核家園, 核四, 核能, 核電, 發電, 核四廠, 廢核, 再生, 除役
                Sample texts from this topic:
                - 能源 政策。
                - 實施 綠色 電價 制度 人民 選舉 綠色 能源 權力 電力 解嚴 打破 台電 能源 壟斷 地位 修改 電業法 推動 電力 自由化
                - 為了 促進 臺灣 能源 加速 轉型 早日 非核 脫油 國際 能源 價格 飆漲 不至於 吃掉 未來 經濟 成長 成果 因此 主張 核一 三廠 提前 除役 核四 絕不 商轉 再生 能源 ２０１９年 發電量 總發電量 ５％ 成立 能源 世代 轉換 監督 委員會 國安 層級 總統 督導 ９９.４％ 能源 進口 臺灣 進行 能源 結構 轉變 提高 自主 能源 比例 臺灣 ２０２５年 再生 能源 發電 總發電 比例 ２０％
                Explanation: Let's think step by step. First, the keywords are aimed at [全民]. Secondly, The sample texts often discuss energy policy and and are not limited to specific regions. Finally, all residents of Taiwan stand to benefit from these proposals. Therefore, the target audience for this topic is [全民].
                Target audience: [全民]
                -----
                Keywords: 運動, 公園, 中心, 親子, 遊具, 共融, 休閒, 設施, 爭取, 體育
                Explanation: The keywords indicate that this topic aimed at construction of sports centers or activity centers which only benefit the residents nearby. Therefore, the target audience for this topic is [地區居民(父母家長親子; 兒童)].
                Target audience: [地區居民(父母家長親子; 兒童)]
                -----
                Keywords: 運動, 公園, 中心, 親子, 遊具, 共融, 休閒, 設施, 爭取, 體育
                Sample texts from this topic:
                - 爭取 經費 增設 全民 運動 中心 打造 特色 公園
                - 爭取 國民 運動 中心
                - 休閒 運動 公園
                Explanation: Let's think step by step. First, the keywords are aimed at [地區居民(父母家長親子; 兒童)]. Secondly, the sample texts aim to advocate for the construction of sports centers or activity centers, which are related to parent-child activities. Finally, based on the information provided, constructing a sports center in a particular area will primarily benefit only the local residents of that area rather than a wide, unspecified majority. This is because nationals from other areas would need to incur additional costs to use this sports center.
                Therefore, [地區居民(父母家長親子), 兒童] is incorrect since it includes 兒童 which is seen as a national wide group. In summary, we categorize the target audience for this topic as local residents, specifically benefiting 父母家長親子, thus it should be labeled as [地區居民(父母家長親子; 兒童)].
                Target audience: [地區居民(父母家長親; 兒童)]
                -----
                Keywords: 衛武營, 鳳山, 高雄, 黃埔, 公園, 藝術, 藝文, 博物館, 文化, 南方
                Explanation: The keywords indicate that this topic aimed at the art and culture industry in Kaohsiung City. Therefore, the target audience for this theme should be [地區居民(公益團體(文化團體))].
                Target audience: [地區居民(公益團體(文化團體))]
                -----
                Keywords: 衛武營, 鳳山, 高雄, 黃埔, 公園, 藝術, 藝文, 博物館, 文化, 南方
                Sample texts from this topic:
                - 衛武營 流行 音樂 中心 時代 來臨 強化 在地 文化 產業 扶植 高雄 藝文 專業 人才
                - 文化 高雄 藝文 城市 打造 高雄 成為 流行 文化 舉辦 藝術 文化節 提倡 藝文 表演 推動 演唱會 經濟 藝文 產業 深耕 扶植 動漫 電競 產業 鼓山 鹽埕 金新興 苓雅 成為 藝術 亮點 串聯 衛武營 文化 中心 中央 公園 海音 中心 駁二 哈瑪星 美術館 內惟 藝術 中心 凹子 森林 公園 創造 藝術 文化 經濟 活化 公有 閒置 場域 推展 親子 婦幼 長輩 青年 活動 空間 發展 高雄 經濟
                - 打造 文化 鳳山 中央 協助 電影 造景 模式 再現 黃埔 新村 風華 打造 唯一 眷村 文化 博物館 推動 五甲 國宅 變身 吸引 藝術家 進駐 成為 衛武營 兩廳院 商業 人才 支援 系統 力爭 政府 扶植 南方 藝術 表演 人才 大東 園區 形成 藝術 黃金 三角 聚落 鳳山 文化 偉大
                Explanation: Let's think step by step. First, the keywords are aimed at  [地區居民(公益團體(文化團體))]. Secondly, the sample texts aim to adcovate for the art and culture industry in Kaohsiung City, along with related professionals, public welfare groups, and youth. Finally, people from other regions of Taiwan will not benefit from these proposals; only the local residents of Kaohsiung will. Therefore, the target audience for this topic is [地區居民(工商企業; 公益團體(文化團體); 青年; 專業人士)].
                Target audience: [地區居民(工商企業; 公益團體(文化團體); 青年; 專業人士)]
                -----

                The topic keywords: {keywords}
                Sample texts from this topic:
                {documents}

                Based on the information above, provide a list of the potential audience that the documents within the topic trying to target and return to the following format, need no provided the explanation.
                Return format: Target audience: [group1, group2, ...]
                """

        # Get response from AI model
        response = get_model_response(client, model_name, prompt, config)
        target_audience, explanation = response.split(":")[1].strip().strip("[]").replace('"', '').split(", "), response.split(":")[0].strip()
        target_audiences.append(target_audience)
        explanations.append(explanation)
        time.sleep(8) # to prevent RateLimitErrors

    # Add results to DataFrame
    df[f'Target_Audience_{model_name}'] = target_audiences
    if model_name in {'GPT', 'Claude'}:
        df[f'{model_name}_Explanation'] = explanations

    return df


def str_to_list(string) -> list:
    if type(string) == str:
        try:
            return ast.literal_eval(string)
        except ValueError:
            return string.strip("[]").replace('\'', '').split(", ")
    elif type(string) == list:
        return string

def check_only_local_residents(lst, audience_list):
    filtered_lst = [item for item in lst if item.startswith('地區居民') or item in audience_list]
    return all(item.startswith('地區居民') for item in filtered_lst) if filtered_lst else False

def similarity_check_(df: pd.DataFrame) -> int:
    gpt_claude = df['PORK_GPT'] == df['PORK_Claude']
    # similarity = (df['PORK_GPT'] == df['PORK_Gemini']) & (df['PORK_GPT'] == df['PORK_Claude']) & (df['PORK_Gemini'] == df['PORK_Claude'])
    print(f"Similarity of GPT vs. Claude: {round(gpt_claude.sum() / len(df), 3)}")

    return gpt_claude.sum()

def classification_by_audience(df: pd.DataFrame, audience_list: list) -> pd.DataFrame:
    df['Target_Audience_GPT'] = df['Target_Audience_GPT'].apply(str_to_list)
    df['Target_Audience_Claude'] = df['Target_Audience_Claude'].apply(str_to_list)
    df['PORK_GPT'] = df['Target_Audience_GPT'].apply(lambda x: 1 if check_only_local_residents(x, audience_list) else 0)
    df['PORK_Claude'] = df['Target_Audience_Claude'].apply(lambda x: 1 if check_only_local_residents(x, audience_list) else 0)

    similarity_check_(df)

    df['PORK_AI'] = np.where(
        df['PORK_GPT'] == df['PORK_Claude'],
        df['PORK_Claude'], # True
        -1 # False: require human feedback
    )

    return df

def calculate_weighted_topic(df_main: pd.DataFrame, df_distribution: pd.DataFrame, df_classification: pd.DataFrame, version: int) -> pd.DataFrame:
    """
    計算加權主題並更新主資料集。

    參數:
        df_main (pd.DataFrame): 主要資料集，包含 ID 欄位。
        df_distribution (pd.DataFrame): 文件的主題分佈資料集，包含 ID 和主題欄位。
        df_classification (pd.DataFrame): 每個主題的二分類結果，每行代表一個主題。

    返回:
        pd.DataFrame: 更新後的主資料集，新增了 Topic_version 欄位。
    """
    # 篩選出主題分佈欄位
    # 將 df_distribution 的列名轉換為整數類型（如果可能）
    df_distribution.columns = [int(col) if col.isdigit() else col for col in df_distribution.columns]

    # 篩選出主題分佈欄位
    topics = [col for col in df_distribution.columns if isinstance(col, int)]

    # 將 df_classification 轉換為適合乘法運算的格式
    df_classification = df_classification.set_index('Topic')['PORK']

    # 計算加權主題分佈
    df_weighted = df_distribution[topics].multiply(df_classification, axis=1)

    # 計算每個文件的加權主題總和
    df_distribution[f'PORK_{version}'] = df_weighted.sum(axis=1)

    # 合併計算結果到主資料集
    df_main = df_main.merge(df_distribution[['ID', f'PORK_{version}']], on='ID')

    return df_main

def similarity_check(df: pd.DataFrame) -> int:
    gpt = df['PORK_GPT'] == df['PORK']
    claude = df['PORK_Claude'] == df['PORK']
    gpt_claude = df['PORK_GPT'] == df['PORK_Claude']
    print(f"Similarity of GPT vs. Claude: {round(gpt_claude.sum() / len(df), 3)}")
    print(f"Similarity of GPT vs. human: {round(gpt.sum() / len(df), 3)}")
    print(f"Similarity of Claude vs. human: {round(claude.sum() / len(df), 3)}")

    return gpt.sum(), claude.sum()

def plot_change(df: pd.DataFrame, columns: list, model='LDA', save=False):
    """
    Plots the mean and standard deviation of specified columns in a DataFrame for each unique threshold (TH) value.

    Args:
    - df (pd.DataFrame): The DataFrame containing the data to plot.
    - columns (list): A list of column names to include in the calculations. These columns should contain numeric data.
    - model (str): The name of the model, used for labeling the plot and saving files. Default is 'LDA'.
    - save (bool): If True, the plot will be saved to a file instead of being displayed. Default is False.

    Description:
    - The function first calculates the mean and standard deviation of the specified columns for each unique 'TH' value in the DataFrame.
    - It then creates a bar plot where each bar represents the mean value for a specific 'TH', with an error bar representing the standard deviation.
    - The plot includes annotations of the mean values above each bar.
    - The X-axis labels are customized based on the 'TH' values, and the plot is titled with the model name.
    - If 'save' is True, the plot is saved to a specified directory with a filename that includes the model name.
    """
    colors = ['#6E7F80', '#A4BE8C', '#D1B2A5']  # More colors can be added for future expansion
    fig, ax = plt.subplots(figsize=(8, 6))

    width_ = 0.3
    means = []
    stds = []
    th_labels = []

    for th in sorted(df['TH'].unique()):
        df_filtered = df[df['TH'] == th]
        mean_value = df_filtered[columns].mean(axis=1)
        mean = mean_value.mean()
        std = mean_value.std()
        print(f"mean: {round(mean, 3)}, std: {round(std, 3)}")

        means.append(mean)
        stds.append(std)
        th_labels.append(f"{th}\n({str(th)})")

    positions = np.arange(len(means))
    bars = ax.bar(positions, means, width=width_, color=colors[:len(means)], capsize=5)

    # Add mean values above each bar
    for bar, mean in zip(bars, means):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, round(mean, 2), ha='center', va='bottom')

    ax.set_xlabel('Election')
    ax.set_ylabel('Average Pork Ratio')
    plt.xticks(positions, th_labels)
    plt.legend(['Mean of Pork Ratio'], loc='best')

    plt.tight_layout()
    if save:
        plt.savefig(f'Graph/Large/Average_Weights_by_Election_{model}.png')
    else:
        plt.show()

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
    """
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
        total = len(df)
        pork_ratio = df['PORK'].mean()
        print(f"PORK: {df['PORK'].sum()}, TOTAL: {df.shape[0]}")
        policy_ratio = 1 - pork_ratio
        ratios.append((pork_ratio, policy_ratio))

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#6E7F80', '#A4BE8C']

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


def perform_t_test(df: pd.DataFrame, columns: list, candidate: str):
    """
    Conduct T test of 6th and 7th election and plot the results including confidence intervals.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    columns (list): List of columns representing different pork ratios.
    candidate (str): Name of the candidate or group being tested.
    """
    mean_values_th1 = df[df['TH'] == 6][columns].mean(axis=1)
    mean_values_th2 = df[df['TH'] == 7][columns].mean(axis=1)
    if stats.levene(mean_values_th1, mean_values_th2)[1] < 0.05:
        print("變異數有顯著差異")
        equal = False
    else:
        equal = True

    alternative = 'two-sided' if candidate == '小黨候選人' else 'less'

    # Perform t-test
    result = ttest_ind(mean_values_th1, mean_values_th2, equal_var=equal, alternative=alternative)

    # Calculate means and standard errors of the mean (SEM) for plotting
    mean1, mean2 = np.mean(mean_values_th1), np.mean(mean_values_th2)
    std1, std2 = np.std(mean_values_th1), np.std(mean_values_th2)

    print(result)
    print(result.confidence_interval(confidence_level=0.95))

    print(f"\n第六屆平均: {round(mean1, 3)}, 第六屆標準差: {round(std1, 3)} \n第七屆平均: {round(mean2, 3)}, 第七屆標準差: {round(std2, 3)} ")
    if result[1] < 0.05:
        print(f"\n{candidate}間存在顯著差異。")
    else:
        print(f"\n{candidate}間不存在顯著差異。")

    return result

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
        
def visualization(df: pd.DataFrame, model: str, party: str, y_axis: float, columns: list, reform_year=2005, highlight=False, save=False, avg=False):
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
    colors = ['#6E7F80', '#D9AF6B', '#A4BE8C', '#D1B2A5', '#957DAD']
    highlight_color = "red"
    averages = []

    if avg:
        df['avg_all_columns'] = df[columns].mean(axis=1)
        overall_avg = df.groupby('TH')['avg_all_columns'].mean()
        overall_avg.plot(ax=ax, label='Average Pork Ratio', color=colors[0], marker='o', linestyle='-', linewidth=2)

        result_df = pd.DataFrame({
            'TH': overall_avg.index,
            'PORK': overall_avg.values,
            })
        print(result_df)

        if highlight:
            # Highlight the section between TH=6 and TH=7
            ax.plot([6, 7], overall_avg.loc[6:7], color=highlight_color, marker='o', linestyle='-', linewidth=2)
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
            print(result_df)

    # ax.set_title(f'Evolution of Pork Ratios in {party} Candidates Across Elections')
    ax.set_xlabel('Election Year')
    ax.axvline(x=6.25, color='grey', linestyle='--')
    ax.annotate('Electoral Reform', xy=(6.25, y_axis), xytext=(6.5, y_axis + 0.05), arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    th_year_dict = {2: "1992", 3: "1995", 4: "1998", 5: "2001", 6: "2004", 7: "2008", 8: "2012", 9: "2016", 10: "2020", 11: "2024"}
    ax.set_xticks(np.arange(2, len(th_year_dict) + 2))
    new_xtick_labels = [f"{th}\n({year})" for th, year in th_year_dict.items()]
    ax.set_xticklabels(new_xtick_labels, fontsize=10)
    ax.set_ylabel('Average Pork Ratio', fontsize=12)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max)  
    ax.set_yticks(np.arange(np.floor(y_min * 20) / 20, np.ceil(y_max * 20) / 20, 0.05))

    if avg:
        ax.legend(loc='best', fontsize=10)
    else:
        lines, labels = ax.get_legend_handles_labels()
        legend_labels = [f'Version {i}' for i in range(1, len(columns) + 1)]
        ax.legend(lines, legend_labels, loc='best')

    if save:
        if avg:
            plt.savefig(f'Graph/Large/{country}_Pork_{model}_{party}_avg.png')
        else:
            plt.savefig(f'Graph/Large/{country}_Pork_{model}_{party}.png')
    else:
        plt.show()

def visualization_dot_plot(df: pd.DataFrame, model: str, columns: list, party: str, save=False):
    """
    Visualization function using dot plot to show each candidate's pork-barrel legislation ratio per election term ('TH'),
    with hollow dots and a line showing the average pork ratio per term.

    Args:
    - df (pd.DataFrame): DataFrame containing the results.
    - model (str): Model name used as part of the file name for saving.
    - columns (list): List of column names containing pork-barrel legislation ratios to be visualized.
    - party (str): Name of the party.
    - reform_year (int): The year of electoral reform, used to annotate the plot.
    - save (bool): Whether to save the plot as a file.
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))
    df['avg_all_columns'] = df[columns].mean(axis=1)

    pre_reform_color = 'black'
    post_reform_color = 'dimgrey'
    for key, grp in df.groupby('TH'):
        color = pre_reform_color if key < 7 else post_reform_color
        ax.scatter([key] * len(grp), grp['avg_all_columns'], alpha=0.6, edgecolors=color, facecolors='none', s=100)

    term_averages = df.groupby('TH')['avg_all_columns'].mean()
    ax.plot(term_averages.index, term_averages, color='#6E7F80', marker='o', linestyle='-', linewidth=2, label='Term Average')
    ax.set_xlabel('Election Year')
    ax.axvline(x=6.25, color='grey', linestyle='-')
    ax.annotate('Electoral Reform', xy=(6.25, 0.92), xytext=(6.25 - 1.75, 0.94), arrowprops=dict(facecolor='black', arrowstyle='->', linewidth=1))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')

    th_year_dict = {2: "1992", 3: "1995", 4: "1998", 5: "2001", 6: "2004", 7: "2008", 8: "2012", 9: "2016", 10: "2020", 11: "2024"}
    ax.set_xticks(np.arange(2, len(th_year_dict) + 2))
    new_xtick_labels = [f"{th}\n({year})" for th, year in th_year_dict.items()]
    ax.set_xticklabels(new_xtick_labels, fontsize=10)
    ax.set_ylabel('Pork Ratio', fontsize=12)
    ax.set_ylim(0, 1)

    if save:
        plt.savefig(f'Graph/Large/{model}_{party}_Pork_Ratios_Dot_Plot.png')
    else:
        plt.show()