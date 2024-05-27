# Example with arguments:
# python evaluate_embeddings_n.py --evaluation_file evaluation_request_12_dossiers_no_requests.json --embedding_provider local_embeddings --embedding_author GroNLP --embedding_function bert-base-dutch-cased --collection_name 12_dossiers_no_requests --vector_db_folder ./vector_stores/12_dossiers_no_requests_chromadb_1024_256_local_embeddings_GroNLP/bert-base-dutch-cased
# python evaluate_embeddings_n.py --evaluation_file evaluation_request_12_dossiers_no_requests.json --embedding_provider local_embeddings --embedding_author meta-llama --embedding_function Meta-Llama-3-8B --collection_name 12_dossiers_no_requests --vector_db_folder ./vector_stores/12_dossiers_no_requests_chromadb_1024_256_local_embeddings_meta-llama/Meta-Llama-3-8B

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
import pandas as pd
from argparse import ArgumentParser
from common.querier import Querier


def main():
    parser = ArgumentParser()
    parser.add_argument("-e", "--evaluation_file", type=str)
    parser.add_argument("-p", "--embedding_provider", type=str)
    parser.add_argument("-a", "--embedding_author", type=str)
    parser.add_argument("-f", "--embedding_function", type=str)
    parser.add_argument("-c", "--collection_name", type=str)
    parser.add_argument("-v", "--vector_db_folder", type=str)

    args = parser.parse_args()
    if (
        args.evaluation_file
        and args.embedding_provider
        and args.embedding_author
        and args.embedding_function
        and args.collection_name
        and args.vector_db_folder
    ):
        evaluation_file = args.evaluation_file
        embedding_provider = args.embedding_provider
        embedding_author = args.embedding_author
        embedding_function = args.embedding_function
        complete_embedding_function = f"{embedding_author}/{embedding_function}"
        collection_name = args.collection_name
        vector_db_folder = args.vector_db_folder
    else:
        print(
            "Please provide the source folder of documents, the output folder name, and the database directory.",
            flush=True,
        )
        exit()

    with open(f"./evaluation/{evaluation_file}", "r") as file:
        evaluation = json.load(file)

    # If vector store folder does not exist, stop
    if not os.path.exists(vector_db_folder):
        print(
            '[Error] ~ There is no vector database for this folder yet. First run "python ingest.py"'
        )
        exit()

    querier = Querier()
    querier.make_chain(collection_name, vector_db_folder)

    # Determine file paths
    csv_file_path = f'./evaluation/results/{evaluation_file.split("/")[-1].replace(".json", "")}_{collection_name.replace("_part_1", "")}_{embedding_function}_request.csv'
    json_file_path = f'./evaluation/results/{evaluation_file.split("/")[-1].replace(".json", "")}_{collection_name.replace("_part_1", "")}_{embedding_function}_request_raw.json'
    last_index = -1

    # Check if csv file exists
    csv_file_exists = os.path.exists(csv_file_path)
    csv_file = open(csv_file_path, "a")
    csv_writer = None

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

    for index, (key, value) in enumerate(evaluation.items()):
        if index <= last_index:
            print(f"Skipping index {index}", flush=True)
            continue
        if not value.get("pages"):
            print("No pages found in the JSON file", flush=True)
            continue
        if not value.get("documents"):
            print("No documents found in the JSON file", flush=True)
            continue
        if not value.get("dossier"):
            print("No dossiers found in the JSON file", flush=True)
            continue

        documents = querier.get_documents_with_scores(key)

        retrieved_page_ids = []
        retrieved_dossier_ids = []
        scores = []
        for document, score in documents:
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
            "page_id": "N/A",
            "dossier_id": value["dossier"][0],
            "retrieved_page_ids": ", ".join(retrieved_page_ids),
            "retrieved_dossier_ids": ", ".join(retrieved_dossier_ids),
            "scores": ", ".join(scores),
            "number_of_correct_dossiers": retrieved_dossier_ids.count(
                value["dossier"][0]
            ),
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

    loc = f'{evaluation_file.split(".")[0]}_{collection_name}_{embedding_function}_request.csv'
    result.to_csv(f"evaluation/results/{loc}")


if __name__ == "__main__":
    main()
