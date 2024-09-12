"""
Example with arguments:
python evaluate_embeddings.py --evaluation_file minbzk.json --embedding_model GroNLP/bert-base-dutch-cased --collection_name minbzk_no_requests --vector_db_folder final_vector_stores_minbzk/minbzk_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased --results_path final_evaluation_minbzk/results --evaluation_directory ./final_evaluation_minbzk
python evaluate_embeddings.py --evaluation_file minbzk_paraphrase.json --embedding_model GroNLP/bert-base-dutch-cased --collection_name minbzk_no_requests --vector_db_folder final_vector_stores_minbzk/minbzk_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased --results_path final_evaluation_minbzk/results --evaluation_directory ./final_evaluation_minbzk
python evaluate_embeddings.py --evaluation_file minbzk_keywords.json --embedding_model GroNLP/bert-base-dutch-cased --collection_name minbzk_no_requests --vector_db_folder final_vector_stores_minbzk/minbzk_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased --results_path final_evaluation_minbzk/results --evaluation_directory ./final_evaluation_minbzk
python evaluate_embeddings.py --evaluation_file minbzk_real_words.json --embedding_model GroNLP/bert-base-dutch-cased --collection_name minbzk_no_requests --vector_db_folder final_vector_stores_minbzk/minbzk_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased --results_path final_evaluation_minbzk/results --evaluation_directory ./final_evaluation_minbzk
python evaluate_embeddings.py --evaluation_file minbzk.json --embedding_model GroNLP/bert-base-dutch-cased --collection_name minbzk_no_requests_stem_stopwords --vector_db_folder final_vector_stores_minbzk/minbzk_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased --results_path final_evaluation_minbzk/results --evaluation_directory ./final_evaluation_minbzk
python evaluate_embeddings.py --evaluation_file minbzk_paraphrase.json --embedding_model GroNLP/bert-base-dutch-cased --collection_name minbzk_no_requests_stem_stopwords --vector_db_folder final_vector_stores_minbzk/minbzk_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased --results_path final_evaluation_minbzk/results --evaluation_directory ./final_evaluation_minbzk
python evaluate_embeddings.py --evaluation_file minbzk_keywords.json --embedding_model GroNLP/bert-base-dutch-cased --collection_name minbzk_no_requests_stem_stopwords --vector_db_folder final_vector_stores_minbzk/minbzk_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased --results_path final_evaluation_minbzk/results --evaluation_directory ./final_evaluation_minbzk
python evaluate_embeddings.py --evaluation_file minbzk_real_words.json --embedding_model GroNLP/bert-base-dutch-cased --collection_name minbzk_no_requests_stem_stopwords --vector_db_folder final_vector_stores_minbzk/minbzk_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased --results_path final_evaluation_minbzk/results --evaluation_directory ./final_evaluation_minbzk
python evaluate_embeddings.py --evaluation_file minbzk.json --embedding_model GroNLP/bert-base-dutch-cased --collection_name minbzk_no_requests_real_words --vector_db_folder final_vector_stores_minbzk/minbzk_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased --results_path final_evaluation_minbzk/results --evaluation_directory ./final_evaluation_minbzk
python evaluate_embeddings.py --evaluation_file minbzk_paraphrase.json --embedding_model GroNLP/bert-base-dutch-cased --collection_name minbzk_no_requests_real_words --vector_db_folder final_vector_stores_minbzk/minbzk_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased --results_path final_evaluation_minbzk/results --evaluation_directory ./final_evaluation_minbzk
python evaluate_embeddings.py --evaluation_file minbzk_keywords.json --embedding_model GroNLP/bert-base-dutch-cased --collection_name minbzk_no_requests_real_words --vector_db_folder final_vector_stores_minbzk/minbzk_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased --results_path final_evaluation_minbzk/results --evaluation_directory ./final_evaluation_minbzk
python evaluate_embeddings.py --evaluation_file minbzk_real_words.json --embedding_model GroNLP/bert-base-dutch-cased --collection_name minbzk_no_requests_real_words --vector_db_folder final_vector_stores_minbzk/minbzk_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased --results_path final_evaluation_minbzk/results --evaluation_directory ./final_evaluation_minbzk
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

import json
from argparse import ArgumentParser
from common import chroma
from common import embeddings as emb
from common.csv_writer import CSVWriter
from common.register_time import Timer
from itertools import islice


def run_embeddings(vector_store, evaluation, evaluation_file, collection_name, results_path, embedding_model, timer):
    # Initialize CSV Writer Object
    csv_writer = CSVWriter(collection_name, embedding_model, evaluation_file=evaluation_file, folder_name=results_path)

    # Find starting index
    start_index = csv_writer.last_index + 1
    if start_index > 0:
        print(f"[Info] ~ Skipping until index {start_index - 1}.", flush=True)
        
    print(f"[Timer] ~ Checkpoint 4: {timer.get_current_duration()}", flush=True)
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

        documents = chroma.get_documents_with_scores(vector_store, key)

        retrieved_page_ids = []
        retrieved_dossier_ids = []
        scores = []

        for document, score in documents:
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
                value["dossier"][0],
                ", ".join(retrieved_page_ids),
                ", ".join(retrieved_dossier_ids),
                ", ".join(scores),
                retrieved_dossier_ids.count(value["dossier"][0]),
                *(retrieved_dossier_ids[i] == value["dossier"][0] for i in range(20)),
            ]
        )
        timer.update_time()
        print(f"[Info] ~ Results written on index: {index}.", flush=True)
    print(f"[Timer] ~ Checkpoint 5: {timer.get_current_duration()}", flush=True)
    csv_writer.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("--evaluation_file", type=str, required=True)
    parser.add_argument("--embedding_model", type=str, required=True)
    parser.add_argument("--collection_name", type=str, required=True)
    parser.add_argument("--vector_db_folder", type=str, required=True)
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--evaluation_directory", type=str, required=True)
    args = parser.parse_args()

    # Read evaluation file
    with open(f"{args.evaluation_directory}/{args.evaluation_file}", "r") as file:
        evaluation = json.load(file)
    print(f"[Info] ~ Number of documents in evaluation: {len(evaluation)}", flush=True)

    # If vector store folder does not exist, stop
    if not os.path.exists(args.vector_db_folder):
        raise ValueError('There is no vector database for this folder yet. First run "ingest.py" for the right dataset.')

    # Initializing Timer
    timer = Timer(args.collection_name, args.embedding_model, evaluation_file=args.evaluation_file, folder_name=args.results_path)

    # Load embeddings and corresponding vector store
    print(f"[Timer] ~ Checkpoint 1: {timer.get_current_duration()}", flush=True)
    embeddings = emb.getEmbeddings(args.embedding_model)
    print(f"[Timer] ~ Checkpoint 2: {timer.get_current_duration()}", flush=True)
    vector_store = chroma.get_chroma_vector_store(args.collection_name, embeddings, args.vector_db_folder)
    print(f"[Timer] ~ Checkpoint 3: {timer.get_current_duration()}", flush=True)

    run_embeddings(vector_store, evaluation, args.evaluation_file, args.collection_name, args.results_path, args.embedding_model, timer)


if __name__ == "__main__":
    main()
