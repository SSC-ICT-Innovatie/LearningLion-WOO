"""
Examples with arguments:
python evaluate_embeddings_document_similarity.py --documents_directory ./docs --results_path ./evaluation/results --embedding_model GroNLP/bert-base-dutch-cased --collection_name 12_dossiers_no_requests --vector_db_folder ./vector_stores/12_dossiers_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased
python evaluate_embeddings_document_similarity.py --documents_directory ./docs_ministries_full --results_path ./evaluation_ministries_full/results --embedding_model GroNLP/bert-base-dutch-cased --collection_name minaz_no_requests --vector_db_folder ./vector_stores/minaz_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased
"""

import os
from dotenv import load_dotenv
from huggingface_hub import login
from sys import platform

load_dotenv()
login(os.environ.get("HUGGINGFACE_API_TOKEN"))
if platform == "linux":
    os.environ["HF_HOME"] = "/scratch/nju"
    os.environ["HF_HUB_CACHE"] = "/scratch/nju"
    os.environ["TRANSFORMERS_CACHE"] = "/scratch/nju"

import pandas as pd
from argparse import ArgumentParser
from common import chroma
from common import embeddings as emb
from common.csv_writer import CSVWriter
from common.register_time import Timer


def run_embeddings_document_similarity(woo_data, vector_store, collection_name, results_path, embedding_model, timer):
    # Initialize CSV Writer Object
    csv_writer = CSVWriter(collection_name, embedding_model, document_similarity=True, folder_name=results_path)

    # Find starting index
    start_index = csv_writer.last_index + 1
    if start_index > 0:
        print(f"[Info] ~ Skipping until index {start_index - 1}.", flush=True)

    for index, row in woo_data.iloc[start_index:].iterrows():
        if pd.isna(row["bodyText"]):
            continue

        documents = chroma.get_documents_with_scores(vector_store, row["bodyText"])

        retrieved_page_ids = []
        retrieved_dossier_ids = []
        scores = []

        for document, score in documents:
            if document.metadata["page_id"] == row["page_id"]:
                # print("[Info] ~ Same document retrieved", flush=True)
                continue
            if document.metadata["page_id"] in retrieved_page_ids:
                # print("[Info] ~ Duplicate page found, skipping.", flush=True)
                continue
            if len(retrieved_page_ids) == 20:
                # print("[Info] ~ 20 documents retrieved", flush=True)
                break
            retrieved_page_ids.append(document.metadata["page_id"])
            retrieved_dossier_ids.append(document.metadata["dossier_id"])
            scores.append(str(score))

        if len(retrieved_page_ids) != 20:
            print(f"[Warning] ~ Only {len(retrieved_page_ids)} retrieved.")

        csv_writer.write_row(
            [
                "N/A",
                row["dossier_id"],
                ", ".join(retrieved_page_ids),
                ", ".join(retrieved_dossier_ids),
                ", ".join(scores),
                retrieved_dossier_ids.count(row["dossier_id"]),
                *(retrieved_dossier_ids[i] == row["dossier_id"] for i in range(20)),
            ]
        )
        timer.update_time()
        print(f"[Info] ~ Results written on index: {index}.", flush=True)
    csv_writer.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("--documents_directory", required=True, type=str)
    parser.add_argument("--embedding_model", required=True, type=str)
    parser.add_argument("--collection_name", required=True, type=str)
    parser.add_argument("--vector_db_folder", required=True, type=str)
    parser.add_argument("--results_path", type=str, required=True)
    args = parser.parse_args()

    # Selecting the paths
    input_path = f"{args.documents_directory}/{args.collection_name}/woo_merged.csv.gz"
    woo_data = pd.read_csv(input_path, compression="gzip")

    # If vector store folder does not exist, stop
    if not os.path.exists(args.vector_db_folder):
        raise ValueError('There is no vector database for this folder yet. First run "ingest.py" for the right dataset.')

    # Initializing Timer
    timer = Timer(args.collection_name, args.embedding_model, document_similarity=True, folder_name=args.results_path)

    embeddings = emb.getEmbeddings(args.embedding_model)
    vector_store = chroma.get_chroma_vector_store(args.collection_name, embeddings, args.vector_db_folder)

    run_embeddings_document_similarity(woo_data, vector_store, args.collection_name, args.results_path, args.embedding_model, timer)


if __name__ == "__main__":
    main()
