"""
Examples with arguments:
python evaluate_bm25_document_similarity.py --algorithm BM25Fast --content_folder_name minaz_no_requests --documents_directory /scratch/nju/docs_ministries_full --results_path /scratch/nju/evaluation_ministries_full/results
"""

import csv
import heapq
import os
import pandas as pd
from argparse import ArgumentParser
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from common import evaluate_helpers
from common.fastbm25 import FastBM25


def run_bm25_document_similarity(woo_data, bm25, results_path, content_folder_name):
    print(f"[Info] ~ Running algorithm: {bm25.__class__.__name__}", flush=True)

    # Determine file paths and check existence
    csv_file_path = f"{results_path}/document_similarity_{content_folder_name}_{bm25.__class__.__name__}.csv"
    last_index = -1

    # If the file exists, determine the last index processed
    if os.path.exists(csv_file_path):
        with open(csv_file_path, "r") as file:
            reader = csv.reader(file)
            last_index = sum(1 for row in reader)

    # Open CSV file for appending or writing
    csv_file = open(csv_file_path, "a", newline="")
    csv_writer = csv.writer(csv_file)

    # Write header if the file is new
    if last_index == -1:
        csv_writer.writerow(
            [
                "page_id",
                "dossier_id",
                "retrieved_page_ids",
                "retrieved_dossier_ids",
                "scores",
                "number_of_correct_dossiers",
                *(f"dossier#{i+1}" for i in range(20)),
            ]
        )

    for index, (_, row) in enumerate(woo_data.iterrows()):
        if index <= last_index:
            print(f"[Info] ~ Skipping index {index}", flush=True)
            continue
        if pd.isna(row["bodyText"]):
            continue
        tokenized_query = evaluate_helpers.tokenize(row["bodyText"])

        # Check if running fastbm25 or not, because FastBM25 is from a different package
        if bm25.__class__.__name__ == "FastBM25":
            n_pages_result = bm25.top_k_sentence_index(tokenized_query, k=21)
        else:
            doc_scores = bm25.get_scores(tokenized_query)
            n_pages_result = heapq.nlargest(21, range(len(doc_scores)), key=doc_scores.__getitem__)

        retrieved_page_ids = []
        retrieved_dossier_ids = []
        # scores = []

        for i in n_pages_result:
            if woo_data["page_id"][i] == row["page_id"]:
                print("[Info] ~ Same document retrieved", flush=True)
                continue
            if len(retrieved_page_ids) == 20:
                print("[Info] ~ 20 documents retrieved", flush=True)
                break
            retrieved_page_ids.append(woo_data["page_id"][i])
            retrieved_dossier_ids.append(woo_data["dossier_id"][i])

        # Append the new row to the DataFrame
        csv_writer.writerow(
            [
                row["page_id"],
                row["dossier_id"],
                ", ".join(retrieved_page_ids),
                ", ".join(retrieved_dossier_ids),
                "",
                retrieved_dossier_ids.count(row["dossier_id"]),
                *((retrieved_dossier_ids[i] == row["dossier_id"] if i < len(retrieved_dossier_ids) else False) for i in range(20)),  # Dossiers 1-20
            ]
        )
        print(f"[Info] ~ Results written on index: {index}.", flush=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--content_folder_name", type=str, required=True)
    parser.add_argument("--documents_directory", type=str, required=True)
    parser.add_argument("--results_path", type=str, required=True)
    args = parser.parse_args()

    content_folder_name = args.content_folder_name
    documents_directory = args.documents_directory
    results_path = args.results_path
    if args.algorithm in ["BM25Okapi", "BM25L", "BM25Plus", "BM25Fast"]:
        algorithm = args.algorithm

    # Selecting the paths
    input_path = f"{documents_directory}/{content_folder_name}/woo_merged.csv.gz"
    woo_data = pd.read_csv(input_path, compression="gzip")

    # Generate corpus, which is a list of all the words per document
    corpus = woo_data["bodyText"].tolist()

    print(f"[Info] ~ Number of documents in corpus: {len(corpus)}", flush=True)

    # Do preprocessing for each document
    tokenized_corpus = [evaluate_helpers.preprocess_text(doc) for doc in corpus]

    if algorithm == "BM25Okapi":
        bm25 = BM25Okapi(tokenized_corpus)
    elif algorithm == "BM25L":
        bm25 = BM25L(tokenized_corpus)
    elif algorithm == "BM25Plus":
        bm25 = BM25Plus(tokenized_corpus)
    elif algorithm == "BM25Fast":
        bm25 = FastBM25(tokenized_corpus)
    else:
        raise ValueError("Algorithm must be one of: BM25Okapi, BM25L, BM25Plus, BM25Fast.")

    print(f"[Info] ~ Starting {algorithm}", flush=True)
    run_bm25_document_similarity(woo_data, bm25, results_path, content_folder_name)
    print(f"[Info] ~ {algorithm} done", flush=True)


if __name__ == "__main__":
    main()
