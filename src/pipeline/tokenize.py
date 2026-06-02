"""
Tokenize manifesto sentences using CKIP Tagger.

Corresponds to: Notebooks/tokenization.ipynb
Requires: CKIP model data at <project_root>/CKIP_TAGGER/ (download separately)
Input:  data/merge_data/{year}.xlsx
Output: data/Dataset/{year}.csv  and  data/Manifesto_Dataset_Origin.csv
"""

import argparse
import os
import pandas as pd

from utils.utils_token import split_content, tokenization, postprocess_dataframe

YEARS = [1992, 1995, 1998, 2001, 2004, 2008, 2012, 2016, 2020, 2024]


def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize manifesto data using CKIP Tagger.")
    parser.add_argument("--data_dir", type=str, default="data", help="Root data directory.")
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=YEARS,
        help="Election years to process (default: all).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir

    dfs = {}
    num = []
    for year in args.years:
        path = os.path.join(data_dir, "merge_data", f"{year}.xlsx")
        dfs[year] = pd.read_excel(path)
        print(dfs[year].shape)
        num.append(dfs[year].shape[0])

    # Split each manifesto into individual policy sentences
    for k in list(dfs.keys()):
        dfs[k] = split_content(dfs[k])

    total = sum(v.shape[0] for v in dfs.values())
    for i, (k, v) in enumerate(dfs.items()):
        t = v.shape[0]
        print(f"{k} 年的選舉共有 {t} 筆資料，候選人平均提出 {round(t / num[i], 2)} 筆政見")
    print(f"一共有 {total} 筆資料")

    # Tokenize (CPU-intensive: ~9 min per year)
    for k in list(dfs.keys()):
        dfs[k] = tokenization(k, dfs[k])

    # Post-processing
    for k in list(dfs.keys()):
        dfs[k] = postprocess_dataframe(dfs[k])

    total_after = sum(v.shape[0] for v in dfs.values())
    print(f"After post-processing: {total_after} records")

    # Save per-year CSVs
    dataset_dir = os.path.join(data_dir, "Dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    for k, v in dfs.items():
        out_path = os.path.join(dataset_dir, f"{k}.csv")
        v.to_csv(out_path, encoding='utf-8-sig', index=False)
        print(f"Saved {out_path}")

    # Save combined CSV
    combined = pd.concat(dfs.values(), ignore_index=True)
    print(combined.shape)
    combined_path = os.path.join(data_dir, "Manifesto_Dataset_Origin.csv")
    combined.to_csv(combined_path, encoding='utf-8-sig', index=False)
    print(f"Saved combined dataset to {combined_path}")


if __name__ == "__main__":
    main()
