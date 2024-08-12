
"""
This script reads in a dataset, extracts only the real words based on this list:
https://github.com/OpenTaal/opentaal-wordlist/blob/master/elements/wordlist-ascii.txt
And then saves it in the same format.

Example with arguments:
python preprocess_real_words.py --content_folder_name minbzk --documents_directory docs_ministries
"""

import os
import pandas as pd
from argparse import ArgumentParser
from common import evaluate_helpers


def filter_body_text(text, words_set):
    # Split text into words and filter them
    filtered_words = [word for word in text.split() if word in words_set]
    # Join the filtered words back into a string
    return ' '.join(filtered_words)


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

    # Only keep real words
    woo_data['bodyText'] = woo_data['bodyText'].apply(lambda x: filter_body_text(str(x), words_set))


    new_file_name = args.content_folder_name + "_real_words"
    save_directory = os.path.join(args.documents_directory, new_file_name)
    if os.path.exists(save_directory):
        raise FileExistsError(f"[Error] ~ The directory '{save_directory}' already exists.", flush=True)
    os.makedirs(save_directory)
    output_path = os.path.join(save_directory, "woo_merged.csv.gz")
    woo_data.to_csv(output_path, compression="gzip", index=False)
    print(f"[Info] ~ Processed data saved to {output_path}.")

if __name__ == "__main__":
    main()
