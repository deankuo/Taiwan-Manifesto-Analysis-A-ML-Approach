"""
Topic modeling at the full-manifesto level using LDA.

Corresponds to: Notebooks/taiwan_analysis_policy.ipynb
Unit of analysis: full candidate manifesto (one row per candidate)

Requires:
  - data/Manifesto_Dataset_Origin.csv
  - API keys: OPENAI_API_KEY, CLAUDE_API_KEY (in .env or environment)

Output: output/Result_v{n}/ with topic_info and result CSVs, figure/ plots
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

from utils.utils_topic_modeling import (
    plot_grid_search_result,
    document_topic_matrix,
    get_combined_topic_info,
    topic_distribution,
    show_topic_keyword,
    generate_target_audience,
    classification_by_audience,
    calculate_weighted_topic,
    similarity_check,
    plot_change,
    perform_t_test,
    visualization,
    visualization_dot_plot,
)

AUDIENCE_LIST = [
    "全民", "軍公教", "台商", "老人", "婦女", "原住民", "族群(閩南、客家族群、眷村)", "外籍人士",
    "學生", "中壯年", "青年", "兒童", "榮民", "勞工", "藝文人士", "工商企業", "醫療人員", "病人", "選手",
    "公益團體(社福團體)", "專業人士", "社工員", "自由行旅客", "特殊技能人士",
    "弱勢(含性工作者、更生人、卡奴、腳踏車騎士)", "僑民", "殘障(身心障礙)", "失業", "低收入戶",
    "中間選民", "投資者", "父母家長親子", "單親家庭", "選任公務人員(議員、里長)", "農漁民", "網民", "地區居民",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Manifesto-level LDA analysis.")
    parser.add_argument("--data_file", type=str, default="data/Manifesto_Dataset_Origin.csv")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--figure_dir", type=str, default="figure")
    parser.add_argument("--version", type=int, default=1, help="Result version number.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_topics", type=int, default=90, help="Final LDA topic count (after grid search).")
    parser.add_argument("--save", action="store_true", help="Save figures.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)

    SEED = args.seed
    np.random.seed(SEED)

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    df = pd.read_csv(args.data_file)
    print(f"Loaded {len(df)} candidates.")

    # Vectorize TOKEN column (manifesto-level)
    data = df.TOKEN.values.tolist()
    vectorizer = CountVectorizer(analyzer='word', max_df=0.99, min_df=int(0.005 * len(data)))
    data_vectorized = vectorizer.fit_transform(data)
    print(f"Vocabulary size: {data_vectorized.shape}")

    # Initial LDA to check baseline
    lda_init = LatentDirichletAllocation(
        n_components=79, max_iter=12, learning_method='batch',
        batch_size=256, learning_decay=0.7, random_state=SEED, evaluate_every=-1, verbose=3,
    )
    lda_init.fit_transform(data_vectorized)
    print("Initial Log Likelihood:", lda_init.score(data_vectorized))
    print("Initial Perplexity:", lda_init.perplexity(data_vectorized))

    # Hyperparameter grid search
    n_topics_range = list(range(89, 95, 1))
    search_params = {'n_components': n_topics_range, 'learning_decay': [.3], 'batch_size': [128]}
    lda_base = LatentDirichletAllocation(max_iter=10, learning_method='batch', random_state=SEED, evaluate_every=-1)
    grid_model = GridSearchCV(lda_base, param_grid=search_params, cv=3, verbose=3)
    grid_model.fit(data_vectorized)

    print("Best Params:", grid_model.best_params_)
    print("Best Log-Likelihood:", grid_model.best_score_)
    print("Best Perplexity:", grid_model.best_estimator_.perplexity(data_vectorized))
    plot_grid_search_result(grid_model, save=args.save)

    # Final LDA model
    best_n = grid_model.best_params_['n_components']
    lda_model = LatentDirichletAllocation(
        n_components=best_n, max_iter=12, learning_method='batch',
        batch_size=128, learning_decay=0.3, random_state=SEED, evaluate_every=-1, verbose=3,
    )
    lda_model.fit_transform(data_vectorized)
    print("Log Likelihood:", lda_model.score(data_vectorized))
    print("Perplexity:", lda_model.perplexity(data_vectorized))
    pprint(lda_model.get_params())

    # Document-topic matrix
    df_document_topic, topicnames = document_topic_matrix(df_=df, lda_model=lda_model, data=data_vectorized)
    topic_info = get_combined_topic_info(vectorizer, lda_model, df_document_topic, df)
    df_topic_distribution = topic_distribution(df_document_topic)
    print(df_topic_distribution)
    print(show_topic_keyword(vectorizer, lda_model, n_keywords=15))

    generation_config = {
        "candidate_count": 1,
        "max_output_tokens": 256,
        "temperature": 0.95,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.1,
    }

    topic_info = generate_target_audience(topic_info, 'GPT', config=generation_config)
    topic_info = generate_target_audience(topic_info, 'Claude', config=generation_config)
    topic_info = classification_by_audience(topic_info, AUDIENCE_LIST)

    result_dir = os.path.join(args.output_dir, "LDA_Large", f"Result_v{args.version}")
    os.makedirs(result_dir, exist_ok=True)
    df_document_topic.to_csv(
        os.path.join(result_dir, f"document_topic_v{args.version}.csv"),
        encoding='utf-8-sig', index=False,
    )
    topic_info.to_csv(
        os.path.join(result_dir, f"topic_info_v{args.version}.csv"),
        encoding='utf-8-sig', index=False,
    )

    # --- Post-classification (requires human-reviewed topic_info CSVs) ---
    versions = 3
    MODEL = 'LDA'
    FILE_PATH = 'LDA_Large'

    topic_info_dict = {}
    df_document_topic_dict = {}
    df_result = df.copy()

    for n in range(1, versions + 1):
        version = f'v{n}'
        ti = pd.read_csv(os.path.join(args.output_dir, FILE_PATH, f"Result_{version}", f"topic_info_{version}.csv"))
        dt = pd.read_csv(os.path.join(args.output_dir, FILE_PATH, f"Result_{version}", f"document_topic_{version}.csv"))
        topic_info_dict[version] = ti
        df_document_topic_dict[version] = dt
        df_result = calculate_weighted_topic(df_result, dt, ti, n)
        print(similarity_check(ti))

    df_serious = df_result[df_result['SERIOUS_CANDIDATE'] == 1]
    cols = ['PORK_v1', 'PORK_v2', 'PORK_v3']
    df_result['PORK_LDA_Manifesto'] = df_result[cols].mean(axis=1)

    # Pre/post reform paired candidates
    df_result['LNAME'] = df_result['LNAME'].astype(str).apply(lambda x: x.split('_')[0])
    filtered_df = df_result[df_result['TH'].isin([6, 7])]
    candidates_in_both = filtered_df.groupby('LNAME').filter(lambda x: x['TH'].nunique() == 2)
    print(candidates_in_both.groupby('TH').size())

    plot_change(candidates_in_both, cols, model=MODEL, save=args.save)

    for party_label, query, y_axis in [
        ('Serious', None, 0.2),
        ('KMT', 'PARTY_CODE == 1', 0.21),
        ('DPP', 'PARTY_CODE == 2', 0.2),
        ('Non-KMT&DPP', 'PARTY_CODE > 2', 0.18),
    ]:
        subset = df_serious if query is None else df_serious.query(query)
        visualization(df=subset, model=MODEL, y_axis=y_axis, party=party_label,
                      columns=cols, reform_year=2005, save=args.save, avg=True)
        visualization_dot_plot(df=subset, model=MODEL, party=party_label,
                               columns=cols, save=args.save)

    for subset_label, subset_df in [
        ('連任候選人', candidates_in_both),
        ('認真型候選人', df_serious),
        ('國民黨候選人', df_serious[df_serious['PARTY_CODE'] == 1]),
        ('民進黨候選人', df_serious[df_serious['PARTY_CODE'] == 2]),
        ('小黨候選人', df_serious[df_serious['PARTY_CODE'] > 2]),
    ]:
        perform_t_test(df=subset_df, columns=cols, candidate=subset_label)


if __name__ == "__main__":
    main()
