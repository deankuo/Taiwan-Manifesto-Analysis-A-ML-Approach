import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils.utils import party_plot
from utils.utils_topic_modeling import (
    plot_change,
    plot_pork_policy_ratios,
    visualization_dot_plot,
    visualization,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot graphs from Peng-Ting Kuo's master's thesis.")
    parser.add_argument("--data_file",
                        type=str,
                        default="data/Dataset.csv",
                        help="Dataset CSV with PORK_* columns.")
    parser.add_argument("--model_number",
                        type=int,
                        default=1,
                        help="1 = LDA Manifesto, 2 = BERTopic EQUAL, 3 = LDA EQUAL")
    parser.add_argument("--output_dir",
                        type=str,
                        default="figure/",
                        help="Output directory for saved figures.")
    parser.add_argument("--model",
                        type=str,
                        default="BERTopic",
                        help="Model name (LDA or BERTopic).")
    parser.add_argument("--height",
                        type=float,
                        default=0.2,
                        help="Arrow height for the reform annotation.")
    parser.add_argument("--party",
                        type=int,
                        default=1,
                        help="Party code (0 = all). See PARTY_CODE in utils.py.")
    parser.add_argument("--save",
                        action="store_true",
                        help="Save figures instead of showing them.")
    parser.add_argument("--avg",
                        action="store_true",
                        help="Average across versions before plotting.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.data_file is not None:
        extension = args.data_file.split(".")[-1]
        assert extension in ["csv", "json"], "`data_file` should be a csv or a json file only."

    df = pd.read_csv(args.data_file)

    if args.party:
        condition = f"PARTY_CODE == {args.party}"
        party = df[df['PARTY_CODE'] == args.party]['PARTY'].iloc[0]
    else:
        condition = "PARTY_CODE > 0"
        party = "All"

    if args.model_number == 1:
        column = ["PORK_LDA_Manifesto"]
    elif args.model_number == 2:
        column = ["PORK_BERTopic_EQUAL"]
    elif args.model_number == 3:
        column = ["PORK_LDA_EQUAL"]
    else:
        raise ValueError(f"Invalid model_number {args.model_number}. Must be 1, 2, or 3.")

    visualization(
        df=df.query(condition),
        model=args.model,
        y_axis=args.height,
        party=party,
        columns=column,
        reform_year=2005,
        highlight=False,
        save=args.save,
        avg=args.avg,
    )

    visualization_dot_plot(
        df=df.query(condition),
        model=args.model,
        party=party,
        columns=column,
        save=args.save,
    )


if __name__ == "__main__":
    main()
