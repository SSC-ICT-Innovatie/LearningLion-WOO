"""
This script reads in a dataset, and stems the words and removes dutch stopwords.
And then saves it in the same format.

Example with arguments:
python preprocess_stem_stopwords.py --content_folder_name minbzk_no_requests --documents_directory final_docs_minbzk --timer_directory final_times
"""

import os
import pandas as pd
from argparse import ArgumentParser
from common import evaluate_helpers
from common.register_time import Timer

def main():
    parser = ArgumentParser()
    parser.add_argument("--content_folder_name", type=str, required=True)
    parser.add_argument("--documents_directory", type=str, required=True)
    parser.add_argument("--timer_directory", type=str, required=True)
    args = parser.parse_args()

    # Selecting the paths
    input_path = os.path.join(args.documents_directory, args.content_folder_name, "woo_merged.csv.gz")
    woo_data = pd.read_csv(input_path, compression="gzip")

    timer = Timer(args.content_folder_name, "preprocess_stem_stopwords", preprocess=True, folder_name=args.timer_directory)

    # Filter text
    woo_data['bodyText'] = woo_data['bodyText'].apply(lambda x: evaluate_helpers.preprocess_text(str(x)))

    new_file_name = args.content_folder_name + "_stem_stopwords"
    save_directory = os.path.join(args.documents_directory, new_file_name)
    if os.path.exists(save_directory):
        raise FileExistsError(f"[Error] ~ The directory '{save_directory}' already exists.", flush=True)
    os.makedirs(save_directory)
    output_path = os.path.join(save_directory, "woo_merged.csv.gz")
    woo_data.to_csv(output_path, compression="gzip", index=False)

    timer.update_time()
    print(f"[Info] ~ Processed data saved to {output_path}.")

if __name__ == "__main__":
    main()
