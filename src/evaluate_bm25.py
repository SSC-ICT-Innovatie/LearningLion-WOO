"""
Example with arguments:
python evaluate_bm25.py --algorithm BM25Okapi --content_folder_name WoogleDumps_01-04-2024_10_dossiers_no_requests_fake_stopwords --documents_directory ../docs --evaluation_directory ./evaluation --evaluation_file evaluation_request_WoogleDumps_01-04-2024_10_dossiers_no_requests.json --results_path ./evaluation/results
python evaluate_bm25.py --algorithm BM25Okapi --content_folder_name 12_dossiers_no_requests --documents_directory ./docs --evaluation_directory ./evaluation --evaluation_file evaluation_request_12_dossiers_no_requests.json --results_path ./evaluation/results
python evaluate_bm25.py --algorithm BM25Okapi --content_folder_name 12_dossiers_no_requests --documents_directory ./docs --evaluation_directory ./evaluation --evaluation_file evaluation_request_60_dossiers_no_requests.json --results_path ./evaluation/results
"""

import heapq
import json
import nltk
import pandas as pd
from argparse import ArgumentParser
from common import evaluate_helpers
from common.csv_writer import CSVWriter
from common.fastbm25 import FastBM25
from common.register_time import Timer
from itertools import islice
from rank_bm25 import BM25Okapi, BM25L, BM25Plus


def run_bm25(woo_data, bm25, evaluation, evaluation_file, content_folder_name, results_path, timer):
    # Initialize CSV Writer Object
    csv_writer = CSVWriter(content_folder_name, bm25.__class__.__name__, evaluation_file=evaluation_file, folder_name=results_path)

    # Find starting index
    start_index = csv_writer.last_index + 1
    if start_index > 0:
        print(f"[Info] ~ Skipping until index {start_index - 1}.", flush=True)

    for index, (key, value) in enumerate(islice(evaluation.items(), start_index, None)):
        if not value.get("pages"):
            print("[Warning] ~ No pages found in the JSON file", flush=True)
            continue
        if not value.get("documents"):
            print("[Warning] ~ No documents found in the JSON file", flush=True)
            continue
        if not value.get("dossier"):
            print("[Warning] ~ No dossiers found in the JSON file", flush=True)
            continue

        tokenized_query = evaluate_helpers.preprocess_text(key)
        # tokenized_query = evaluate_helpers.tokenize(key)

        # Check if running FastBM25 or not, because FastBM25 uses different code
        if bm25.__class__.__name__ == "FastBM25":
            doc_scores = bm25.top_scores(tokenized_query)
            n_pages_result = heapq.nlargest(21, doc_scores, key=doc_scores.__getitem__)
        else:
            doc_scores = bm25.get_scores(tokenized_query)
            n_pages_result = heapq.nlargest(21, range(len(doc_scores)), key=doc_scores.__getitem__)

        retrieved_page_ids = []
        retrieved_dossier_ids = []
        scores = []

        for i in n_pages_result:
            retrieved_page_ids.append(woo_data["page_id"][i])
            retrieved_dossier_ids.append(woo_data["dossier_id"][i])
            scores.append(doc_scores[i])

        csv_writer.write_row(
            [
                "N/A",
                value["dossier"][0],
                ", ".join(retrieved_page_ids),
                ", ".join(retrieved_dossier_ids),
                ", ".join(map(str, scores)),
                retrieved_dossier_ids.count(value["dossier"][0]),
                *((retrieved_dossier_ids[i] == value["dossier"][0] if i < len(retrieved_dossier_ids) else False) for i in range(20)),
            ]
        )
        timer.update_time()
        print(f"[Info] ~ Results written on index: {index}.", flush=True)
    csv_writer.close()


def main():
    # If necessary, download the NLTK resources
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    parser = ArgumentParser()
    parser.add_argument("--algorithm", type=str, choices=["BM25Okapi", "BM25L", "BM25Plus", "BM25Fast"], required=True)
    parser.add_argument("--content_folder_name", type=str, required=True)
    parser.add_argument("--documents_directory", type=str, required=True)
    parser.add_argument("--evaluation_directory", type=str, required=True)
    parser.add_argument("--evaluation_file", type=str, required=True)
    parser.add_argument("--retrieve_whole_document", type=bool, default=False)
    parser.add_argument("--results_path", type=str, required=True)
    args = parser.parse_args()

    # Selecting the paths
    input_path = f"{args.documents_directory}/{args.content_folder_name}/woo_merged.csv.gz"
    evaluation_path = f"{args.evaluation_directory}/{args.evaluation_file}"
    woo_data = pd.read_csv(input_path, compression="gzip")

    # Preprocess woo data, merge all the pages into one document, instead of seperately
    if args.retrieve_whole_document:
        woo_data = (
            woo_data.groupby("document_id")
            .agg(
                {
                    "bodyText": lambda x: " ".join(x.astype(str)),
                    "page_id": "first",
                    "dossier_id": "first",
                }
            )
            .reset_index()
        )

    # Initializing Timer
    timer = Timer(args.content_folder_name, args.algorithm, evaluation_file=args.evaluation_file, folder_name=args.results_path)

    # Generate corpus, which is a list of all the words per document
    corpus = woo_data["bodyText"].tolist()
    print(f"[Info] ~ Number of documents in corpus: {len(corpus)}", flush=True)

    # Do preprocessing for each document
    tokenized_corpus = [evaluate_helpers.preprocess_text(doc) for doc in corpus]

    # Read evaluation File
    with open(evaluation_path, "r") as file:
        evaluation = json.load(file)
    print(f"[Info] ~ Number of documents in evaluation: {len(evaluation)}", flush=True)

    # Choose the appropriate algorithm
    if args.algorithm == "BM25Okapi":
        bm25 = BM25Okapi(tokenized_corpus)
    elif args.algorithm == "BM25L":
        bm25 = BM25L(tokenized_corpus)
    elif args.algorithm == "BM25Plus":
        bm25 = BM25Plus(tokenized_corpus)
    elif args.algorithm == "BM25Fast":
        bm25 = FastBM25(tokenized_corpus)
    else:
        raise ValueError("Algorithm must be one of: BM25Okapi, BM25L, BM25Plus, BM25Fast.")

    run_bm25(woo_data, bm25, evaluation, args.evaluation_file, args.content_folder_name, args.results_path, timer)


if __name__ == "__main__":
    main()
