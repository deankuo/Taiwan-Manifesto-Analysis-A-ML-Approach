"""
Topic modeling at the policy-segment level using BERTopic + LDA.

Corresponds to: Notebooks/taiwan_analysis_segment.ipynb
Unit of analysis: individual policy sentences (split from each manifesto)

Requires:
  - data/Manifesto_Dataset.csv  (sentence-tokenized, combined)
  - Embedding file (OpenAI text-embedding-3-large) saved as CSV
  - API keys: OPENAI_API_KEY, CLAUDE_API_KEY (in .env or environment)

Output: output/Result_v{n}/ with topic_info and result CSVs, figure/ plots
"""

import argparse
import os
import uuid
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import openai

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic.representation import OpenAI as BERTopicOpenAI

from utils.utils_topic_modeling import (
    show_text_length,
    plot_grid_search_result,
    document_topic_matrix,
    get_combined_topic_info,
    topic_distribution,
    generate_target_audience,
    classification_by_audience,
    similarity_check,
    merge_and_classify,
    plot_pork_policy_ratios,
    plot_change,
    perform_t_test,
    visualization_dot_plot,
    visualization,
    OPENAI_API_KEY,
)

YEARS = [1992, 1995, 1998, 2001, 2004, 2008, 2012, 2016, 2020, 2024]
AUDIENCE_LIST = [
    "全民", "軍公教", "台商", "老人", "婦女", "原住民", "族群(閩南、客家族群、眷村)", "外籍人士",
    "學生", "中壯年", "青年", "兒童", "榮民", "勞工", "藝文人士", "工商企業", "醫療人員", "病人", "選手",
    "公益團體(社福團體)", "專業人士", "社工員", "自由行旅客", "特殊技能人士",
    "弱勢(含性工作者、更生人、卡奴、腳踏車騎士)", "僑民", "殘障(身心障礙)", "失業", "低收入戶",
    "中間選民", "投資者", "父母家長親子", "單親家庭", "選任公務人員(議員、里長)", "農漁民", "網民", "地區居民",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Segment-level BERTopic + LDA analysis.")
    parser.add_argument("--data_file", type=str, default="data/Manifesto_Dataset.csv")
    parser.add_argument("--embedding_file", type=str, default="data/Embedding_OPENAI_SENTENCE.csv",
                        help="Pre-computed OpenAI embedding CSV (rows x dims, no header).")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--figure_dir", type=str, default="figure")
    parser.add_argument("--version", type=int, default=1, help="Result version number.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_bertopic", action="store_true", help="Skip BERTopic and run LDA only.")
    parser.add_argument("--save", action="store_true", help="Save figures.")
    return parser.parse_args()


def run_bertopic(df, embeddings, seed, openai_api_key, output_dir, version, save, figure_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.1, metric='cosine', random_state=seed)
    hdbscan_model = HDBSCAN(min_cluster_size=25, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    vectorizer_model = CountVectorizer(max_df=0.99, min_df=0.005)

    summarization_prompt = """
I have a topic that is described by the following keywords: [KEYWORDS]
In this topic, the following documents are a small but representative subset of all documents in the topic:
[DOCUMENTS]

Based on the information above, please give a description of this topic in the following format and use only traditional Chinese with no more than 15 words:
topic: <description>
"""
    client = openai.OpenAI(api_key=openai_api_key)
    gpt_turbo = BERTopicOpenAI(client, model="gpt-4o", delay_in_seconds=4, chat=True,
                               prompt=summarization_prompt, generator_kwargs={'top_p': 0.95, 'temperature': 0.95})
    representation_model = {"GPT": gpt_turbo}

    topic_model = BERTopic(
        language="Chinese",
        verbose=True,
        calculate_probabilities=True,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
    )

    documents = df["SENTENCE"].tolist()
    topics, probs = topic_model.fit_transform(documents, embeddings)

    # Fine-tune representation labels with GPT
    topic_model.update_topics(docs=documents, representation_model=representation_model)
    topic_info_gpt = topic_model.get_topic_info()

    generation_config = {
        "candidate_count": 1,
        "max_output_tokens": 256,
        "temperature": 0.95,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1,
    }

    topic_info = generate_target_audience(df=topic_info_gpt, model_name='GPT', config=generation_config,
                                          unit='policy', BERTopic=True)
    topic_info = generate_target_audience(df=topic_info, model_name='Claude', config=generation_config,
                                          unit='policy', BERTopic=True)
    topic_info = classification_by_audience(topic_info, AUDIENCE_LIST)

    result_dir = os.path.join(output_dir, f"BERTopic", f"Result_v{version}")
    os.makedirs(result_dir, exist_ok=True)
    topic_info.to_csv(os.path.join(result_dir, f"topic_info_v{version}.csv"), encoding='utf-8-sig', index=False)

    topic_column = topic_model.get_document_info(documents)
    df_new = pd.concat([df, topic_column[['Topic']]], axis=1)
    df_new.rename(columns={"Topic": f"Topic_v{version}"}, inplace=True)
    df_new.to_csv(os.path.join(result_dir, f"Result_v{version}.csv"), encoding='utf-8-sig', index=False)

    return topic_info, df_new


def run_lda(df, seed, output_dir, version, save, figure_dir):
    df = df[df['TOKEN'].notna()].reset_index(drop=True)

    num_rows = len(df)
    df['ID_OF_PART'] = [str(uuid.uuid4()) for _ in range(num_rows)]

    data = df.TOKEN.values.tolist()

    vectorizer = CountVectorizer(analyzer='word', max_df=0.99, min_df=int(0.0004 * len(data)))
    data_vectorized = vectorizer.fit_transform(data)
    print(f"Vocabulary size: {data_vectorized.shape}")

    # Hyperparameter search
    n_topics = list(range(152, 166, 1))
    search_params = {'n_components': n_topics, 'learning_decay': [.3], 'batch_size': [128]}
    lda_base = LatentDirichletAllocation(max_iter=10, learning_method='batch', random_state=seed,
                                         evaluate_every=-1, n_jobs=-1)
    grid_model = GridSearchCV(lda_base, param_grid=search_params, cv=3, verbose=3)
    grid_model.fit(data_vectorized)

    print("Best Params:", grid_model.best_params_)
    print("Best Log-Likelihood:", grid_model.best_score_)
    plot_grid_search_result(grid_model, save=save)

    best_n = grid_model.best_params_['n_components']
    lda_model = LatentDirichletAllocation(
        n_components=best_n, max_iter=12, learning_method='batch',
        batch_size=128, learning_decay=0.3, random_state=seed,
        evaluate_every=-1, verbose=3,
    )
    lda_model.fit_transform(data_vectorized)
    print("Log Likelihood:", lda_model.score(data_vectorized))
    print("Perplexity:", lda_model.perplexity(data_vectorized))
    pprint(lda_model.get_params())

    df_document_topic, topicnames = document_topic_matrix(df_=df, lda_model=lda_model, data=data_vectorized)
    topic_info = get_combined_topic_info(vectorizer, lda_model, df_document_topic, df)
    df_topic_distribution = topic_distribution(df_document_topic)
    print(df_topic_distribution)

    generation_config = {
        "candidate_count": 1,
        "max_output_tokens": 256,
        "temperature": 0.95,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1,
    }

    topic_info = generate_target_audience(df=topic_info, model_name='GPT', config=generation_config)
    topic_info = generate_target_audience(df=topic_info, model_name='Claude', config=generation_config)
    topic_info = classification_by_audience(topic_info, AUDIENCE_LIST)

    result_dir = os.path.join(output_dir, "LDA", f"Result_v{version}")
    os.makedirs(result_dir, exist_ok=True)
    topic_info.to_csv(os.path.join(result_dir, f"topic_info_v{version}.csv"), encoding='utf-8-sig', index=False)

    df_new = pd.concat([df, df_document_topic[['ID_OF_PART', 'Dominant_topic']]], axis=1)
    df_new.rename(columns={"Dominant_topic": f"Topic_v{version}"}, inplace=True)
    df_new.to_csv(os.path.join(result_dir, f"Result_v{version}.csv"), encoding='utf-8-sig', index=False)

    return topic_info, df_new, df_document_topic


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # Load data
    df = pd.read_csv(args.data_file)
    df = df[df['SENTENCE'].notna()].reset_index(drop=True)
    df['SENTENCE'] = df['SENTENCE'].astype(str)
    df['CLEAN_CONTENT'] = df['CLEAN_CONTENT'].astype(str)
    df['CONTENT_LENGTH'] = df.groupby('ID')['CONTENT'].transform(lambda x: x.apply(len))

    print(f"Total documents: {len(df)}")
    print(f"Serious candidate documents: {len(df[df['SERIOUS_CANDIDATE'] == 1])}")

    text_length = df.groupby('ID')['CONTENT_LENGTH'].sum().reset_index()
    show_text_length(text_length['CONTENT_LENGTH'], save=args.save)

    # BERTopic (requires pre-computed embeddings)
    if not args.skip_bertopic:
        print("Loading embeddings...")
        embedding_df = pd.read_csv(args.embedding_file, header=None)
        embeddings = embedding_df.to_numpy()
        print(f"Embeddings shape: {embeddings.shape}")
        run_bertopic(df, embeddings, SEED, OPENAI_API_KEY, args.output_dir, args.version, args.save, args.figure_dir)

    # LDA
    topic_info_lda, df_lda, df_document_topic = run_lda(df, SEED, args.output_dir, args.version, args.save, args.figure_dir)

    # --- Post-classification result merging (requires human-reviewed topic_info CSVs) ---
    versions = 3
    MODEL = 'LDA'
    df_origin = pd.read_csv(os.path.join(os.path.dirname(args.data_file), "Manifesto_Dataset_Origin.csv"))
    df_new = pd.read_csv(os.path.join(args.output_dir, MODEL, f"Result_v{versions}", f"Result_v{versions}.csv"))

    topic_info = {}
    for n in range(1, versions + 1):
        version = f'v{n}'
        ti = pd.read_csv(os.path.join(args.output_dir, MODEL, f"Result_{version}", f"topic_info_{version}.csv"))
        topic_info[version] = ti
        pork = ti[ti['PORK'] == 1]['Topic'].tolist()
        df_new[f'PORK_v{n}'] = df_new[f'Topic_v{n}'].apply(lambda x: 1 if x in pork else 0)
        print(similarity_check(ti))

    df_merge = merge_and_classify(df_new, ['v1', 'v2', 'v3'])
    df_merge = pd.merge(df_origin, df_merge, on='ID', how='inner')
    df_merge.fillna(0, inplace=True)
    df_merge['LNAME'] = df_merge['LNAME'].astype(str).apply(lambda x: x.split('_')[0])

    df_serious = df_merge[df_merge['SERIOUS_CANDIDATE'] == 1]
    print(f"Serious candidates: {len(df_serious)}")

    weight_cols = ['WEIGHT_PORK_v1', 'WEIGHT_PORK_v2', 'WEIGHT_PORK_v3']
    equal_divided_cols = ['PART_PORK_v1', 'PART_PORK_v2', 'PART_PORK_v3']

    # Pre/post reform paired candidates
    filtered_df = df_merge[df_merge['TH'].isin([6, 7])]
    candidates_in_both = filtered_df.groupby('LNAME').filter(lambda x: x['TH'].nunique() == 2)

    # Visualizations
    plot_pork_policy_ratios(dfs=[topic_info['v1'], topic_info['v2'], topic_info['v3']],
                            titles=['Version1', 'Version2', 'Version3'], save=args.save)
    plot_change(candidates_in_both, weight_cols, model=MODEL, save=args.save)

    for party_label, query, y_axis in [
        ('Serious', None, 0.1),
        ('KMT', 'PARTY_CODE == 1', 0.1),
        ('DPP', 'PARTY_CODE == 2', 0.12),
        ('Non-KMT&DPP', 'PARTY_CODE > 2', 0.1),
    ]:
        subset = df_serious if query is None else df_serious.query(query)
        visualization(df=subset, model=MODEL, y_axis=y_axis, party=party_label,
                      columns=equal_divided_cols, reform_year=2005, save=args.save, avg=True)
        visualization_dot_plot(df=subset, model=MODEL, party=party_label,
                               columns=equal_divided_cols, save=args.save)

    # T-tests
    for subset_label, subset_df in [
        ('連任候選人', candidates_in_both),
        ('認真型候選人', df_serious),
        ('國民黨候選人', df_serious[df_serious['PARTY_CODE'] == 1]),
        ('民進黨候選人', df_serious[df_serious['PARTY_CODE'] == 2]),
        ('小黨候選人', df_serious[df_serious['PARTY_CODE'] > 2]),
    ]:
        perform_t_test(df=subset_df, columns=weight_cols, candidate=subset_label)


if __name__ == "__main__":
    main()
