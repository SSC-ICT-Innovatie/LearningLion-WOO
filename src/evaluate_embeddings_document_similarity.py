# Examples with arguments:
# python evaluate_embeddings_document_similarity.py --content_folder_name 12_dossiers_no_requests --documents_directory ./docs --embedding_provider local_embeddings --embedding_author GroNLP --embedding_function bert-base-dutch-cased --collection_name 12_dossiers_no_requests --vector_db_folder ./vector_stores/12_dossiers_no_requests_chromadb_1024_256_local_embeddings_GroNLP/bert-base-dutch-cased
# python evaluate_embeddings_document_similarity.py --content_folder_name 12_dossiers_no_requests --documents_directory ./docs --embedding_provider local_embeddings --embedding_author meta-llama --embedding_function Meta-Llama-3-8B --collection_name 12_dossiers_no_requests --vector_db_folder ./vector_stores/12_dossiers_no_requests_chromadb_1024_256_local_embeddings_meta-llama/Meta-Llama-3-8B

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
from common.querier import Querier


def main():
    parser = ArgumentParser()
    parser.add_argument("--content_folder_name", required=True, type=str)
    parser.add_argument("--documents_directory", required=True, type=str)
    parser.add_argument("--embedding_provider", required=True, type=str)
    parser.add_argument("--embedding_author", required=True, type=str)
    parser.add_argument("--embedding_function", required=True, type=str)
    parser.add_argument("--collection_name", required=True, type=str)
    parser.add_argument("--vector_db_folder", required=True, type=str)

    args = parser.parse_args()
    if (
        args.content_folder_name
        and args.documents_directory
        and args.embedding_provider
        and args.embedding_author
        and args.embedding_function
        and args.collection_name
        and args.vector_db_folder
    ):
        content_folder_name = args.content_folder_name
        documents_directory = args.documents_directory
        embedding_provider = args.embedding_provider
        embedding_author = args.embedding_author
        embedding_function = args.embedding_function
        complete_embedding_function = f"{embedding_author}/{embedding_function}"
        collection_name = args.collection_name
        vector_db_folder = args.vector_db_folder
    else:
        raise ValueError("Please provide all arguments.")

    querier = Querier(
        embeddings_provider=embedding_provider,
        embeddings_model=complete_embedding_function,
    )
    querier.make_chain(collection_name, vector_db_folder)

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
    woo_data = pd.read_csv(
        f"./{documents_directory}/{content_folder_name}/woo_merged.csv.gz"
    )
    for _, row in woo_data.iterrows():
        if pd.isna(row["bodyText"]):
            continue
        documents = querier.get_documents_with_scores(row["bodyText"])
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
            "number_of_correct_dossiers": retrieved_dossier_ids.count(
                row["dossier_id"]
            ),
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

    result.to_csv(
        f"evaluation/results/document_similarity_{content_folder_name}_{collection_name}_{embedding_function}.csv"
    )
    print(
        f"[Info] ~ Result embeddings document similarity for {content_folder_name} with {embedding_function} saved.",
        flush=True,
    )


if __name__ == "__main__":
    main()
