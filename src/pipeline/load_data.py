"""
Load, clean, and merge manifesto data with voting records.

Corresponds to: Notebooks/load_data.ipynb
Output: data/merge_data/{year}.xlsx  and  data/manifesto_dataset.xlsx
"""

import argparse
import os
import uuid
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.utils import (
    load_dataframe,
    fill_education,
    fill_age,
    party_plot,
    party_processing,
    content_processing,
    dataset_cleaning,
    is_main_party,
    vote_calculate,
    find_duplicate_lnames,
)

YEARS = [1992, 1995, 1998, 2001, 2004, 2008, 2012, 2016, 2020, 2024]


def parse_args():
    parser = argparse.ArgumentParser(description="Load and merge manifesto + vote data.")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory.")
    parser.add_argument("--save_party_plot", action="store_true", help="Save the party-count plot.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

    # --- Raw manifesto data ---
    dfs = {}
    for year in YEARS:
        path = os.path.join(data_dir, "manifesto_data", f"taiwan_manifesto_{year}.xlsx")
        dfs[year] = pd.read_excel(path)
        dfs[year] = load_dataframe(dfs[year], year)

    find_duplicate_lnames(dfs)

    for year in YEARS:
        dfs[year] = party_processing(dfs[year])
        dfs[year] = content_processing(dfs[year])

    # Party count plot
    unique_counts = [len(dfs[y]['PARTY'].unique()) for y in YEARS]
    party_plot(unique_counts, YEARS, save=args.save_party_plot)

    # --- Vote data ---
    dfs_vote = {}
    for year in YEARS:
        path = os.path.join(data_dir, "vote_data", f"vote_{year}.csv")
        dfs_vote[year] = pd.read_csv(path)
        dfs_vote[year]['LNAME'] = dfs_vote[year]['LNAME'].apply(lambda x: x.strip())

    dfs_vote, education_differences = fill_education(dfs_vote)
    dfs_vote = fill_age(dfs_vote)
    print("Education differences:", education_differences)

    for k, v in dfs_vote.items():
        v.to_csv(
            os.path.join(data_dir, "vote_data", f"vote_{k}.csv"),
            encoding='utf-8-sig',
            index=False,
        )

    # --- Merge ---
    merged = {}
    for year in YEARS:
        merged[year] = pd.merge(dfs[year], dfs_vote[year], on='LNAME', how='outer')

    # 2008 contains by-election candidates — drop rows missing 性別
    merged[2008] = merged[2008].dropna(subset=['性別'])

    # --- Dataset cleaning ---
    for year in YEARS:
        merged[year] = dataset_cleaning(merged[year])

    # --- Candidate classification ---
    for year in YEARS:
        merged[year]['ENOUGH_VOTE'] = merged[year].groupby('AREA', group_keys=False).apply(vote_calculate)
        merged[year]['MAIN_PARTY_MEMBER'] = is_main_party(merged[year])
        merged[year]['SERIOUS_CANDIDATE'] = merged[year][['ENOUGH_VOTE', 'MAIN_PARTY_MEMBER']].any(axis=1).astype(int)
        num_rows = len(merged[year])
        merged[year].insert(0, 'ID', [str(uuid.uuid4()) for _ in range(num_rows)])
        out_path = os.path.join(data_dir, "merge_data", f"{year}.xlsx")
        merged[year].to_excel(out_path, index=False)
        print(f"Saved {out_path}")

    # --- Combined dataset ---
    df_all = pd.concat(list(merged.values())).reset_index(drop=True)
    print(f"Total records: {len(df_all)}")
    out_path = os.path.join(data_dir, "manifesto_dataset.xlsx")
    df_all.to_excel(out_path, index=False)
    print(f"Saved combined dataset to {out_path}")


if __name__ == "__main__":
    main()
