# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

from Utils.utils import party_plot
from Utils.utils_topic_modeling import (
    plot_change,
    plot_pork_policy_ratios,
    visualization_dot_plot,
    visualization
)

def parse_args():
    parser = argparse.ArgumentParser(description="Plot graphs of Peng-Ting Kuo's master's thesis.")
    parser.add_argument("--data_file", 
                        type=str, 
                        default="Data/Dataset.csv",
                        help="The data_file should include PORK_LDA_Manifesto, PORK_BERTopic_WEIGHT, PORK_BERTopic_EQUAL, PORK_LDA_WEIGHT, PORK_LDA_EQUAL columns or more, if the column names are different, specifying them.")
    parser.add_argument("--model_number",
                        type=int,
                        default=1,
                        help="Specify the model number in Peng-Ting Kuo's master's thesis. (ranging from 1-3)")
    parser.add_argument("--output_dir",
                        type=str,
                        default="Graph/",
                        help="Specify an output directory if save.")
    parser.add_argument("--model", 
                        type=str, 
                        default="BERTopic", 
                        help="Specify the model you use, there's only LDA and BERTopic available.")
    parser.add_argument("--height", 
                        type=float, 
                        default=0.2, 
                        help="The height of arrow in the graph.")
    parser.add_argument("--party", 
                        type=int, 
                        default=1, 
                        help="Which party you are interested in. (Entered 0 for all candidates. Please refer to PARTY_CODE for more information)")
    parser.add_argument("--save", 
                        type=bool, 
                        default=False)
    parser.add_argument("--avg", 
                        type=bool, 
                        default=False, 
                        help="Whether to take average of the results or not.")
    args = parser.parse_args()

    # Sanity checks
    if args.data_file is not None:
        extension = args.train_file.split(".")[-1]
        assert extension in ["csv", "json"], "`data_file` should be a csv or a json file only."
    if args.save:
        output_dir = args.output_dir
        assert output_dir is not None

    return args

def main():
    args = parse_args()
    df = pd.read_csv(args.data_file)
    if args.party:
        condition = f"PARTY_CODE == {args.party}"
        party =  df[df['PARTY_CODE'] == args.party]['PARTY'].iloc[0]
    else:
        condition = "PARTY_CODE > 0"
        party = "All"
    
    if args.model_number == 1:
        column = "PORK_LDA_Manifesto"
    elif args.model_number == 2:
        column = "PORK_BERTopic_EQUAL"
    elif args.model_number == 3:
        column = "PORK_LDA_EQUAL"
    else:
        raise NameError
        
    
    visualization(df=df.query(condition),
                  model=args.model,
                  y_axis=args.height,
                  party=party,
                  columns=column,
                  reform_year=2005,
                  highlight=False,
                  save=args.save,
                  avg=args.avg)
    
    visualization_dot_plot(df=df.query(condition),
                  model=args.model,
                  party=party,
                  columns=column,
                  save=args.save
                  )
        

if __name__ == "__main__":
    main()
