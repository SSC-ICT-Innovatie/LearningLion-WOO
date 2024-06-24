"""
Examples with arguments:
python evaluate_bm25_document_similarity.py --algorithm BM25Fast --content_folder_name minaz_no_requests --documents_directory /scratch/nju/docs_ministries_full --results_path /scratch/nju/evaluation_ministries_full/results
"""

import heapq
import pandas as pd
from argparse import ArgumentParser
from common import evaluate_helpers
from common.csv_writer import CSVWriter
from common.fastbm25 import FastBM25
from common.register_time import Timer
from rank_bm25 import BM25Okapi, BM25L, BM25Plus


def run_bm25_document_similarity(woo_data, bm25, results_path, content_folder_name, timer):
    # Initialize CSV Writer Object
    csv_writer = CSVWriter(content_folder_name, bm25.__class__.__name__, document_similarity=True, folder_name=results_path)

    # Find starting index
    start_index = csv_writer.last_index + 1
    if start_index > 0:
        print(f"[Info] ~ Skipping until index {start_index - 1}.", flush=True)

    for index, row in woo_data.iloc[start_index:].iterrows():
        if pd.isna(row["bodyText"]):
            continue

        tokenized_query = evaluate_helpers.preprocess_text(row["bodyText"])
        # tokenized_query = evaluate_helpers.tokenize(row["bodyText"])

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
            if woo_data["page_id"][i] == row["page_id"]:
                # print("[Info] ~ Same document retrieved", flush=True)
                continue
            if len(retrieved_page_ids) == 20:
                # print("[Info] ~ 20 documents retrieved", flush=True)
                break
            retrieved_page_ids.append(woo_data["page_id"][i])
            retrieved_dossier_ids.append(woo_data["dossier_id"][i])
            scores.append(doc_scores[i])

        csv_writer.write_row(
            [
                row["page_id"],
                row["dossier_id"],
                ", ".join(retrieved_page_ids),
                ", ".join(retrieved_dossier_ids),
                ", ".join(map(str, scores)),
                retrieved_dossier_ids.count(row["dossier_id"]),
                *((retrieved_dossier_ids[i] == row["dossier_id"] if i < len(retrieved_dossier_ids) else False) for i in range(20)),
            ]
        )
        timer.update_time()
        print(f"[Info] ~ Results written on index: {index}.", flush=True)
    csv_writer.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("--algorithm", type=str, choices=["BM25Okapi", "BM25L", "BM25Plus", "BM25Fast"], required=True)
    parser.add_argument("--content_folder_name", type=str, required=True)
    parser.add_argument("--documents_directory", type=str, required=True)
    parser.add_argument("--results_path", type=str, required=True)
    args = parser.parse_args()

    # Selecting the paths
    input_path = f"{args.documents_directory}/{args.content_folder_name}/woo_merged.csv.gz"
    woo_data = pd.read_csv(input_path, compression="gzip")

    # Initializing Timer
    timer = Timer(args.content_folder_name, args.algorithm, document_similarity=True, folder_name=args.results_path)

    # Generate corpus, which is a list of all the words per document
    corpus = woo_data["bodyText"].tolist()
    print(f"[Info] ~ Number of documents in corpus: {len(corpus)}", flush=True)

    # Do preprocessing and tokenization for each document
    tokenized_corpus = [evaluate_helpers.preprocess_text(doc) for doc in corpus]

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

    run_bm25_document_similarity(woo_data, bm25, args.results_path, args.content_folder_name, timer)


if __name__ == "__main__":
    main()
