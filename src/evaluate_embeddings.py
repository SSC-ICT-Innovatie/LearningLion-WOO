# Example with arguments:
# python evaluate_embeddings.py --results_path ./evaluation_ministries_full/results --evaluation_directory ./evaluation_ministries_full --evaluation_file evaluation_request_minaz.json --embedding_model GroNLP/bert-base-dutch-cased --collection_name minaz_no_requests --vector_db_folder ./vector_stores/minaz_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased

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

import csv
import json
import pandas as pd
from argparse import ArgumentParser
from common import chroma
from common import embeddings as emb


def main():
    parser = ArgumentParser()
    parser.add_argument("--evaluation_file", type=str, required=True)
    parser.add_argument("--embedding_model", type=str, required=True)
    parser.add_argument("--collection_name", type=str, required=True)
    parser.add_argument("--vector_db_folder", type=str, required=True)
    parser.add_argument("--results_path", type=str, required=True)
    parser.add_argument("--evaluation_directory", type=str, required=True)

    args = parser.parse_args()

    evaluation_file = args.evaluation_file
    embedding_model = args.embedding_model
    embedding_function = embedding_model.split("/")[-1]
    collection_name = args.collection_name
    vector_db_folder = args.vector_db_folder
    results_path = args.results_path
    evaluation_directory = args.evaluation_directory

    with open(f"{evaluation_directory}/{evaluation_file}", "r") as file:
        evaluation = json.load(file)

    # If vector store folder does not exist, stop
    if not os.path.exists(vector_db_folder):
        raise ValueError('There is no vector database for this folder yet. First run "python ingest.py"')

    embeddings = emb.getEmbeddings(embedding_model)
    vector_store = chroma.get_chroma_vector_store(collection_name, embeddings, vector_db_folder)

    # Determine file paths
    csv_file_path = os.path.join(results_path, f"{evaluation_file.replace('.json', '')}_{embedding_function}_request.csv")
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

    result = pd.DataFrame(
        columns=[
            "page_id",
            "dossier_id",
            "retrieved_page_ids",
            "retrieved_dossier_ids",
            "scores",
            "number_of_correct_dossiers",
            "dossier#1",
            "dossier#2",
            "dossier#3",
            "dossier#4",
            "dossier#5",
            "dossier#6",
            "dossier#7",
            "dossier#8",
            "dossier#9",
            "dossier#10",
            "dossier#11",
            "dossier#12",
            "dossier#13",
            "dossier#14",
            "dossier#15",
            "dossier#16",
            "dossier#17",
            "dossier#18",
            "dossier#19",
            "dossier#20",
        ]
    )

    print("[Info] ~ Starting with the first item", flush=True)

    for index, (key, value) in enumerate(evaluation.items()):
        if index <= last_index:
            print(f"[Info] ~ Skipping index {index}", flush=True)
            continue
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
                print("[Info] ~ Duplicate page found, skipping.", flush=True)
                continue
            if len(retrieved_page_ids) == 20:
                print("[Info] ~ 20 documents retrieved", flush=True)
                break
            retrieved_page_ids.append(document.metadata["page_id"])
            retrieved_dossier_ids.append(document.metadata["dossier_id"])
            scores.append(str(score))
        if len(retrieved_page_ids) != 20:
            print(f"[Warning] ~ Only {len(retrieved_page_ids)} retrieved.")

        new_row = {
            "page_id": "N/A",
            "dossier_id": value["dossier"][0],
            "retrieved_page_ids": ", ".join(retrieved_page_ids),
            "retrieved_dossier_ids": ", ".join(retrieved_dossier_ids),
            "scores": ", ".join(scores),
            "number_of_correct_dossiers": retrieved_dossier_ids.count(value["dossier"][0]),
            "dossier#1": retrieved_dossier_ids[0] == value["dossier"][0],
            "dossier#2": retrieved_dossier_ids[1] == value["dossier"][0],
            "dossier#3": retrieved_dossier_ids[2] == value["dossier"][0],
            "dossier#4": retrieved_dossier_ids[3] == value["dossier"][0],
            "dossier#5": retrieved_dossier_ids[4] == value["dossier"][0],
            "dossier#6": retrieved_dossier_ids[5] == value["dossier"][0],
            "dossier#7": retrieved_dossier_ids[6] == value["dossier"][0],
            "dossier#8": retrieved_dossier_ids[7] == value["dossier"][0],
            "dossier#9": retrieved_dossier_ids[8] == value["dossier"][0],
            "dossier#10": retrieved_dossier_ids[9] == value["dossier"][0],
            "dossier#11": retrieved_dossier_ids[10] == value["dossier"][0],
            "dossier#12": retrieved_dossier_ids[11] == value["dossier"][0],
            "dossier#13": retrieved_dossier_ids[12] == value["dossier"][0],
            "dossier#14": retrieved_dossier_ids[13] == value["dossier"][0],
            "dossier#15": retrieved_dossier_ids[14] == value["dossier"][0],
            "dossier#16": retrieved_dossier_ids[15] == value["dossier"][0],
            "dossier#17": retrieved_dossier_ids[16] == value["dossier"][0],
            "dossier#18": retrieved_dossier_ids[17] == value["dossier"][0],
            "dossier#19": retrieved_dossier_ids[18] == value["dossier"][0],
            "dossier#20": retrieved_dossier_ids[19] == value["dossier"][0],
        }

        # Append the new value to the DataFrame
        # result.append(new_row, ignore_index=True)
        result.loc[len(result)] = new_row

        csv_writer.writerow(
            [
                "N/A",
                value["dossier"][0],
                ", ".join(retrieved_page_ids),
                ", ".join(retrieved_dossier_ids),
                "",
                retrieved_dossier_ids.count(value["dossier"][0]),
                *(retrieved_dossier_ids[i] == value["dossier"][0] for i in range(20)),
            ]
        )
        print(f"[Info] ~ Results written on index: {index}.", flush=True)


if __name__ == "__main__":
    main()
