# Examples with arguments:
# python evaluate_embeddings_document_similarity.py --content_folder_name minaz_no_requests --documents_directory ./docs_ministries_full --results_path ./evaluation_ministries_full/results --embedding_model GroNLP/bert-base-dutch-cased --collection_name minaz_no_requests --vector_db_folder ./vector_stores/minaz_no_requests_chromadb_1024_256_GroNLP/bert-base-dutch-cased


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
import pandas as pd
from argparse import ArgumentParser
from common import chroma
from common import embeddings as emb


def main():
    parser = ArgumentParser()
    parser.add_argument("--content_folder_name", required=True, type=str)
    parser.add_argument("--documents_directory", required=True, type=str)
    parser.add_argument("--embedding_model", required=True, type=str)
    parser.add_argument("--collection_name", required=True, type=str)
    parser.add_argument("--vector_db_folder", required=True, type=str)
    parser.add_argument("--results_path", type=str, required=True)

    args = parser.parse_args()

    content_folder_name = args.content_folder_name
    documents_directory = args.documents_directory
    embedding_model = args.embedding_model
    embedding_function = embedding_model.split("/")[-1]
    collection_name = args.collection_name
    vector_db_folder = args.vector_db_folder
    results_path = args.results_path

    embeddings = emb.getEmbeddings(embedding_model)
    vector_store = chroma.get_chroma_vector_store(collection_name, embeddings, vector_db_folder)

    # Determine file paths
    csv_file_path = os.path.join(results_path, f"document_similarity_{collection_name}_{embedding_function}.csv")
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
    # First create ground truth
    woo_data = pd.read_csv(f"{documents_directory}/{content_folder_name}/woo_merged.csv.gz")
    for index, (_, row) in enumerate(woo_data.iterrows()):
        if index <= last_index:
            print(f"[Info] ~ Skipping index {index}", flush=True)
            continue
        if pd.isna(row["bodyText"]):
            continue
        documents = chroma.get_documents_with_scores(vector_store, row["bodyText"])
        retrieved_page_ids = []
        retrieved_dossier_ids = []
        scores = []
        for document, score in documents:
            if document.metadata["page_id"] == row["page_id"]:
                print("[Info] ~ Same document retrieved", flush=True)
                continue
            if document.metadata["page_id"] in retrieved_page_ids:
                print("[Info] ~ Duplicate page found, skipping.")
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
            "page_id": row["page_id"],
            "dossier_id": row["dossier_id"],
            "retrieved_page_ids": ", ".join(retrieved_page_ids),
            "retrieved_dossier_ids": ", ".join(retrieved_dossier_ids),
            "scores": ", ".join(scores),
            "number_of_correct_dossiers": retrieved_dossier_ids.count(row["dossier_id"]),
            "dossier#1": retrieved_dossier_ids[0] == row["dossier_id"],
            "dossier#2": retrieved_dossier_ids[1] == row["dossier_id"],
            "dossier#3": retrieved_dossier_ids[2] == row["dossier_id"],
            "dossier#4": retrieved_dossier_ids[3] == row["dossier_id"],
            "dossier#5": retrieved_dossier_ids[4] == row["dossier_id"],
            "dossier#6": retrieved_dossier_ids[5] == row["dossier_id"],
            "dossier#7": retrieved_dossier_ids[6] == row["dossier_id"],
            "dossier#8": retrieved_dossier_ids[7] == row["dossier_id"],
            "dossier#9": retrieved_dossier_ids[8] == row["dossier_id"],
            "dossier#10": retrieved_dossier_ids[9] == row["dossier_id"],
            "dossier#11": retrieved_dossier_ids[10] == row["dossier_id"],
            "dossier#12": retrieved_dossier_ids[11] == row["dossier_id"],
            "dossier#13": retrieved_dossier_ids[12] == row["dossier_id"],
            "dossier#14": retrieved_dossier_ids[13] == row["dossier_id"],
            "dossier#15": retrieved_dossier_ids[14] == row["dossier_id"],
            "dossier#16": retrieved_dossier_ids[15] == row["dossier_id"],
            "dossier#17": retrieved_dossier_ids[16] == row["dossier_id"],
            "dossier#18": retrieved_dossier_ids[17] == row["dossier_id"],
            "dossier#19": retrieved_dossier_ids[18] == row["dossier_id"],
            "dossier#20": retrieved_dossier_ids[19] == row["dossier_id"],
        }

        # Append the new row to the DataFrame
        result.loc[len(result)] = new_row

        csv_writer.writerow(
            [
                "N/A",
                row["dossier_id"],
                ", ".join(retrieved_page_ids),
                ", ".join(retrieved_dossier_ids),
                "",
                retrieved_dossier_ids.count(row["dossier_id"]),
                *(retrieved_dossier_ids[i] == row["dossier_id"] for i in range(20)),
            ]
        )
        print(f"[Info] ~ Results written on index: {index}.", flush=True)


if __name__ == "__main__":
    main()
