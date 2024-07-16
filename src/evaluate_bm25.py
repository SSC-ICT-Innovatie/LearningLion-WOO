import bm25s
import json
import os
import pandas as pd
from argparse import ArgumentParser
from common import evaluate_helpers, bm_25_helpers
from common.csv_writer import CSVWriter
from common.register_time import Timer
from itertools import islice


def main():
    parser = ArgumentParser()
    parser.add_argument("--algorithm", type=str, choices=["BM25S"], required=True)
    parser.add_argument("--bm25_retriever_folder", type=str, required=True)
    parser.add_argument("--content_folder_name", type=str, required=True)
    parser.add_argument("--documents_directory", type=str, required=True)
    parser.add_argument("--evaluation_directory", type=str, required=True)
    parser.add_argument("--evaluation_file", type=str, required=True)
    parser.add_argument("--retrieve_whole_document", type=bool, default=False)
    parser.add_argument("--results_path", type=str, required=True)
    args = parser.parse_args()

    # Selecting the paths
    input_path = os.path.join(args.documents_directory, args.content_folder_name, "woo_merged.csv.gz")
    woo_data = pd.read_csv(input_path, compression="gzip")

    # Read evaluation File
    evaluation_path = os.path.join(args.evaluation_directory, args.evaluation_file)
    with open(evaluation_path, "r") as file:
        evaluation = json.load(file)

    # Initialize CSV Writer and Timer
    csv_writer = CSVWriter(args.content_folder_name, args.algorithm, evaluation_file=args.evaluation_file, folder_name=args.results_path)
    timer = Timer(args.content_folder_name, args.algorithm, ingest=True, folder_name=args.bm25_retriever_folder)

    folder_name = bm_25_helpers.generate_name(args.algorithm, args.content_folder_name)
    save_directory = os.path.join(args.bm25_retriever_folder, folder_name)
    retriever = bm25s.BM25.load(save_directory)

    # Find starting index
    start_index = csv_writer.last_index + 1
    if start_index > 0:
        print(f"[Info] ~ Skipping until index {start_index - 1}.", flush=True)

    for index, (key, value) in enumerate(islice(evaluation.items(), start_index, None)):
        if not value.get("pages"):
            print("[Warning] ~ No pages found in the JSON file.", flush=True)
            continue
        if not value.get("documents"):
            print("[Warning] ~ No documents found in the JSON file.", flush=True)
            continue
        if not value.get("dossier"):
            print("[Warning] ~ No dossiers found in the JSON file.", flush=True)
            continue

        preprocessed_query = evaluate_helpers.preprocess_text_no_stem(key)
        query_tokens = bm25s.tokenize(preprocessed_query)
        results, scores = retriever.retrieve(query_tokens, k=20)

        retrieved_page_ids = []
        retrieved_dossier_ids = []
        retrieved_scores = []

        for result, score in zip(results[0], scores[0]):
            retrieved_page_ids.append(woo_data["page_id"][result])
            retrieved_dossier_ids.append(woo_data["dossier_id"][result])
            retrieved_scores.append(str(score))

        csv_writer.write_row(
            [
                "N/A",
                value["dossier"][0],
                ", ".join(retrieved_page_ids),
                ", ".join(retrieved_dossier_ids),
                ", ".join(retrieved_scores),
                retrieved_dossier_ids.count(value["dossier"][0]),
                *((retrieved_dossier_ids[i] == value["dossier"][0] if i < len(retrieved_dossier_ids) else False) for i in range(20)),
            ]
        )
        timer.update_time()
        print(f"[Info] ~ Results written on index: {index}.", flush=True)
    csv_writer.close()


if __name__ == "__main__":
    main()