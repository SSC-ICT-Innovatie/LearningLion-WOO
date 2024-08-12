"""
This script reads in a dataset, and calculates the percentage of real words based on this list:
https://github.com/OpenTaal/opentaal-wordlist/blob/master/elements/wordlist-ascii.txt
And then saves the number as a float (0-1) in a new column called real_words_percentage.

Example with arguments:
python preprocess_calc_real_words_percentage.py --content_folder_name minbzk --documents_directory docs_ministries
"""

import os
import pandas as pd
from argparse import ArgumentParser
from common import evaluate_helpers


def calculate_real_words_percentage(text, words_set):
    words = text.split()
    if not words:
        return 0.0
    real_words_count = sum(1 for word in words if word in words_set)
    return real_words_count / len(words)


def main():
    parser = ArgumentParser()
    parser.add_argument("--content_folder_name", type=str, required=True)
    parser.add_argument("--documents_directory", type=str, required=True)
    args = parser.parse_args()

    # Selecting the paths
    input_path = os.path.join(args.documents_directory, args.content_folder_name, "woo_merged.csv.gz")
    woo_data = pd.read_csv(input_path, compression="gzip")

    words_set = set()
    with open("./common/stopwords/wordlist-ascii.txt", 'r') as file:
        for line in file:
            words_set.add(line.strip())

    # Filter text
    woo_data['bodyText'] = woo_data['bodyText'].apply(lambda x: evaluate_helpers.preprocess_text_no_stem(str(x)))

    # Calculate percentage of real words
    woo_data['real_words_percentage'] = woo_data['bodyText'].apply(lambda x: calculate_real_words_percentage(str(x), words_set))

    new_file_name = args.content_folder_name + "_real_words_percentage"
    save_directory = os.path.join(args.documents_directory, new_file_name)
    if os.path.exists(save_directory):
        raise FileExistsError(f"[Error] ~ The directory '{save_directory}' already exists.")
    os.makedirs(save_directory)
    output_path = os.path.join(save_directory, "woo_merged.csv.gz")
    woo_data.to_csv(output_path, compression="gzip", index=False)
    print(f"[Info] ~ Processed data saved to {output_path}.")

if __name__ == "__main__":
    main()
