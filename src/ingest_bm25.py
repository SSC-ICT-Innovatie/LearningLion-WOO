"""
Example with arguments:
python evaluate_bm25.py --algorithm BM25Okapi --content_folder_name WoogleDumps_01-04-2024_10_dossiers_no_requests_fake_stopwords --documents_directory ../docs --evaluation_directory ./evaluation --evaluation_file evaluation_request_WoogleDumps_01-04-2024_10_dossiers_no_requests.json --results_path ./evaluation/results
"""

import bm25s
import nltk
import pandas as pd
import os
from argparse import ArgumentParser
from common import evaluate_helpers, bm_25_helpers
from common.register_time import Timer
from common.stopwords.stopwords import dutch_stopwords

def main():
    # If necessary, download the NLTK resources
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    parser = ArgumentParser()
    parser.add_argument("--algorithm", type=str, choices=["BM25S"], required=True)
    parser.add_argument("--content_folder_name", type=str, required=True)
    parser.add_argument("--documents_directory", type=str, required=True)
    parser.add_argument("--bm25_retriever_folder", type=str, required=True)
    args = parser.parse_args()

    # Initializing Timer
    timer = Timer(args.content_folder_name, args.algorithm, ingest=True, folder_name=args.bm25_retriever_folder)

    # Selecting the paths
    input_path = os.path.join(args.documents_directory, args.content_folder_name, "woo_merged.csv.gz")
    woo_data = pd.read_csv(input_path, compression="gzip")

    # Generate corpus, which is a list of all the words per document
    corpus = woo_data["bodyText"].tolist()

    print(f"[Timer] ~ Checkpoint 1; Loading data in pandas: {timer.get_current_duration()}", flush=True)

    # Preprocess corpus
    preprocessed_corpus = [evaluate_helpers.preprocess_text_no_stem(body_text) for body_text in corpus]
    # Tokenize the corpus and only keep the ids (faster and saves memory)
    corpus_tokens = bm25s.tokenize(preprocessed_corpus, stopwords=dutch_stopwords)

    # Create the BM25 model and index the corpus
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    # You can save the arrays to a directory...
    folder_name = bm_25_helpers.generate_name(args.algorithm, args.content_folder_name)
    save_directory = os.path.join(args.bm25_retriever_folder, folder_name)
    if os.path.exists(save_directory):
        raise FileExistsError(f"[Error] ~ The directory '{save_directory}' already exists.", flush=True)
    retriever.save(save_directory)

    print(f"[Timer] ~ Checkpoint 2; Creating retriever: {timer.get_current_duration()}", flush=True)
    timer.update_time()


if __name__ == "__main__":
    main()
