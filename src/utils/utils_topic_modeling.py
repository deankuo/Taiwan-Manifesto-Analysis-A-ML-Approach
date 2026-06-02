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

# sklearn
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from pprint import pprint

# OpenAI
import openai

# Gemini
try:
    import google.generativeai as genai
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False

# Huggingface / torch
import torch
from transformers import pipeline as hf_pipeline

# Claude
import anthropic

# API keys from environment (set in .env or shell before running)
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY', '')
HUGGINGFACE_API_KEY = os.environ.get('HUGGINGFACE_API_KEY', '')

if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
if _GEMINI_AVAILABLE and GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

client_claude = anthropic.Anthropic(api_key=CLAUDE_API_KEY) if CLAUDE_API_KEY else None
client_openai = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Hyperparameters
TEMPERATURE = 0.7
TOP_P = 0.95
TOP_K = 30

# Token count of manifestos' length
def show_text_length(column: list, save=False):
    """
    Shows the distribution of text length.

    Args:
        column (list): Should be a column from a pd.DataFrame
        save (bool, optional): Defaults to False.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(column, bins=30, alpha=0.7, color='grey', edgecolor="black", linewidth=0.5)
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
    Plot mean test scores for each hyperparameter combination in GridSearchCV.
    """
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
    """
    lda_output = lda_model.transform(data)
    topicnames = [f"Topic {i}" for i in range(lda_model.n_components)]
    df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames)
    dominant_topic = np.argmax(df_document_topic.values, axis=1)
    df_document_topic['Dominant_topic'] = dominant_topic
    df_document_topic['ID'] = df_['ID']
    return df_document_topic, topicnames

def get_combined_topic_info(vectorizer: CountVectorizer, lda_model: LatentDirichletAllocation, df_document_topic: pd.DataFrame, df: pd.DataFrame, n_keywords=10, n_docs=3):
    """
    Extracts and combines key information for each topic from an LDA model.
    """
    keywords = np.array(vectorizer.get_feature_names_out())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_keywords]
        topic_keywords.append(keywords.take(top_keyword_locs))

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

    df_final = pd.DataFrame({
        'Topic': range(len(topic_keywords)),
        'Keywords': topic_keywords
    })
    df_final = pd.merge(df_final, df_top_docs)
    df_final = df_final.rename(columns={'TOKEN': 'Representative_Docs'})

    return df_final

# Create Topic Distribution
def topic_distribution(df_document_topic: pd.DataFrame) -> pd.DataFrame:
    df = df_document_topic['Dominant_topic'].value_counts().reset_index(name="Document Count")
    df.columns = ['Topic', 'Document Count']
    return df

def show_topic_keyword(vectorizer: CountVectorizer, lda_model: LatentDirichletAllocation, n_keywords: int = 10) -> pd.DataFrame:
    """
    Displays the top n_keywords for each topic in an LDA model.
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
    """
    if model_name == 'GPT':
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=config['temperature'],
            max_tokens=config['max_output_tokens'],
            top_p=config['top_p'],
        )
        return response.choices[0].message.content.strip()

    elif model_name == 'Claude':
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=config['max_output_tokens'],
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()

    else:
        raise ValueError(f"{model_name} is not a valid model name.")

def generate_target_audience(df: pd.DataFrame, model_name: str, config: dict, unit: str = 'manifesto', BERTopic: bool = False) -> pd.DataFrame:
    """
    Generates target audience profiles based on provided DataFrame using AI models.
    """
    if model_name not in {'GPT', 'Gemini', 'Taiwan-llama', 'Claude'}:
        raise ValueError(f"The {model_name} model is not included!")

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

                The topic keywords: {keywords}
                Sample texts from this topic:
                {documents}

                Based on the information above, provide a list of the potential audience that the documents within the topic trying to target and return to the following format, need no provided the explanation.
                Return format: Target audience: [group1, group2, ...]
                """

        response = get_model_response(client, model_name, prompt, config)
        target_audience, explanation = response.split(":")[1].strip().strip("[]").replace('"', '').split(", "), response.split(":")[0].strip()
        target_audiences.append(target_audience)
        explanations.append(explanation)
        time.sleep(8)

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
        df['PORK_Claude'],
        -1
    )

    return df

def calculate_weighted_topic(df_main: pd.DataFrame, df_distribution: pd.DataFrame, df_classification: pd.DataFrame, version: int) -> pd.DataFrame:
    """
    計算加權主題並更新主資料集。
    """
    df_distribution.columns = [int(col) if col.isdigit() else col for col in df_distribution.columns]
    topics = [col for col in df_distribution.columns if isinstance(col, int)]
    df_classification = df_classification.set_index('Topic')['PORK']
    df_weighted = df_distribution[topics].multiply(df_classification, axis=1)
    df_distribution[f'PORK_{version}'] = df_weighted.sum(axis=1)
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
    Plots the mean and standard deviation of pork ratios for TH 6 vs 7.
    """
    colors = ['#6E7F80', '#A4BE8C', '#D1B2A5']
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
    topic_info = pd.read_csv(f'Result/Result_{topic_v}/topic_info_{topic_v}.csv')
    pork = topic_info[topic_info['PORK'] == 1]['Topic'].tolist()
    df[f'PORK_{topic_v}'] = df['Topic'].apply(lambda x: 1 if x in pork else 0)
    return df

def merge_and_classify(df: pd.DataFrame, version_names: list) -> pd.DataFrame:
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
    Plot stacked bar chart showing pork vs. policy ratio per model version.
    """
    width = 0.3
    ratios = []
    for df in dfs:
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
    ax.legend(loc='best')
    plt.tight_layout()
    if save:
        plt.savefig('Graph/Pork_vs_Policy_Topics_Ratio_by_Version.png')
    else:
        plt.show()


def perform_t_test(df: pd.DataFrame, columns: list, candidate: str):
    """
    Conduct T test of 6th and 7th election and plot results.
    """
    from scipy.stats import shapiro, mannwhitneyu

    mean_values_th1 = df[df['TH'] == 6][columns].mean(axis=1)
    mean_values_th2 = df[df['TH'] == 7][columns].mean(axis=1)

    stat1, p1 = shapiro(mean_values_th1)
    stat2, p2 = shapiro(mean_values_th2)
    print(f"Shapiro-Wilk Test: Stat= {stat1} , p-value= {p1}")
    if p1 < 0.05:
        print("資料可能不是常態分佈")
    print(f"Shapiro-Wilk Test: Stat= {stat2} , p-value= {p2}")
    if p2 < 0.05:
        print("資料可能不是常態分佈")

    if p1 < 0.05 or p2 < 0.05:
        u_stat, u_p = mannwhitneyu(mean_values_th1, mean_values_th2, alternative='less' if candidate != '小黨候選人' else 'two-sided')
        print(f"Mann-Whitney U test: U stat= {u_stat} , p-value= {u_p}")

    if stats.levene(mean_values_th1, mean_values_th2)[1] < 0.05:
        print("變異數有顯著差異")
        equal = False
    else:
        equal = True

    alternative = 'two-sided' if candidate == '小黨候選人' else 'less'
    result = ttest_ind(mean_values_th1, mean_values_th2, equal_var=equal, alternative=alternative)

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
    Line plot of pork ratio trends across elections with reform year annotation.
    """
    print(f'Number of candidates: {len(df)}')
    country = 'Taiwan' if reform_year == 2005 else 'Japan'

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#6E7F80', '#D9AF6B', '#A4BE8C', '#D1B2A5', '#957DAD']
    highlight_color = "red"

    if avg:
        df['avg_all_columns'] = df[columns].mean(axis=1)
        overall_avg = df.groupby('TH')['avg_all_columns'].mean()
        overall_avg.plot(ax=ax, label='Average Pork Ratio', color=colors[0], marker='o', linestyle='-', linewidth=2)

        result_df = pd.DataFrame({'TH': overall_avg.index, 'PORK': overall_avg.values})
        print(result_df)

        if highlight:
            ax.plot([6, 7], overall_avg.loc[6:7], color=highlight_color, marker='o', linestyle='-', linewidth=2)
    else:
        for index, column in enumerate(columns):
            grouped = df.groupby('TH')[column].mean()
            policy_ratios = 1 - grouped
            grouped.plot(ax=ax, label=column, color=colors[index % len(colors)], marker='o', linestyle='-')

            result_df = pd.DataFrame({'TH': grouped.index, 'PORK': grouped.values, 'POLICY': policy_ratios.values})
            print(result_df)

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
            plt.savefig(f'figure/{country}_Pork_{model}_{party}_avg.png')
        else:
            plt.savefig(f'figure/{country}_Pork_{model}_{party}.png')
    else:
        plt.show()

def visualization_dot_plot(df: pd.DataFrame, model: str, columns: list, party: str, save=False):
    """
    Dot plot showing each candidate's pork ratio per election term with term average line.
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
        plt.savefig(f'figure/{model}_{party}_Pork_Ratios_Dot_Plot.png')
    else:
        plt.show()
