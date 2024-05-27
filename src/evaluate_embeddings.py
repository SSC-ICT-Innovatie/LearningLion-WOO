# Example with arguments:
# python evaluate_embeddings.py --evaluation_file evaluation_request_12_dossiers_no_requests.json --embedding_provider local_embeddings --embedding_author GroNLP --embedding_function bert-base-dutch-cased --collection_name 12_dossiers_no_requests --vector_db_folder ./vector_stores/12_dossiers_no_requests_chromadb_1024_256_local_embeddings_GroNLP/bert-base-dutch-cased
# python evaluate_embeddings.py --evaluation_file evaluation_request_12_dossiers_no_requests.json --embedding_provider local_embeddings --embedding_author meta-llama --embedding_function Meta-Llama-3-8B --collection_name 12_dossiers_no_requests --vector_db_folder ./vector_stores/12_dossiers_no_requests_chromadb_1024_256_local_embeddings_meta-llama/Meta-Llama-3-8B

import json
import os
import pandas as pd
from argparse import ArgumentParser
from common.querier import Querier


def check_relevance(ground_truth, retrieved) -> int:
    """
    Calculates the number of relevant items in the retrieved set.

    Parameters:
    ground_truth (set): The set of ground truth items.
    retrieved (set): The set of retrieved items.

    Returns:
    int: The number of relevant items in the retrieved set.
    """
    return len(retrieved.intersection(ground_truth))


def get_first_n_unique_ids_by_type(
    source_documents: list, n: int, id_type: str
) -> list:
    """
    Extracts the first n unique document IDs from a list of source documents.

    Parameters:
    - source_documents: A list of tuples, where each tuple contains a document object and another value.
    - n: The number of unique document IDs to retrieve.

    Returns:
    A list of the first n unique document IDs.
    """

    if id_type not in ["page_id", "document_id", "dossier_id"]:
        raise ValueError("id_type must be 'page_id', 'document_id', or 'dossier_id'")

    unique_ids = []
    seen = set()
    for doc, _ in source_documents:
        doc_id = doc.metadata[id_type]
        if doc_id not in seen:
            seen.add(doc_id)
            unique_ids.append(doc_id)
        if len(unique_ids) == n:
            break

    return unique_ids


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

    # Selecting the paths
    # path = select_woogle_dump_folders(path='../docs')
    # evaluation_file = "../evaluation/evaluation_request_WoogleDumps_01-04-2024_50_dossiers_no_requests.json"
    # embedding_provider = "local_embeddings"
    # embedding_author = "GroNLP"
    # embedding_function = "bert-base-dutch-cased"
    # complete_embedding_function = f"{embedding_author}/{embedding_function}"
    # collection_name = "WoogleDumps_01-04-2024_12817_dossiers_no_requests_part_1"
    # # vector_db_folder = f"./vector_stores/no_requests_all_parts_chromadb_1024_256_local_embeddings_GroNLP/{embedding_function}"
    # vector_db_folder = f"../vector_stores/no_requests_part_2_chromadb_1024_256_local_embeddings_GroNLP/{embedding_function}"

    with open(f"./evaluation/{evaluation_file}", "r") as file:
        evaluation = json.load(file)

    # If vector store folder does not exist, stop
    if not os.path.exists(vector_db_folder):
        print(
            'There is no vector database for this folder yet. First run "python ingest.py"'
        )
        exit()

    querier = Querier()
    querier.make_chain(collection_name, vector_db_folder)

    querier_data = querier.vector_store.get()
    querier_data_ids = querier_data["ids"]
    print(f"Length querier data IDs: {len(querier_data_ids)}", flush=True)
    print(f"Max Id in data: {max([int(num) for num in querier_data_ids])}", flush=True)

    print(f"Running algorithm: {complete_embedding_function}", flush=True)

    # Determine file paths
    csv_file_path = f'./evaluation/results/{evaluation_file.split("/")[-1].replace(".json", "")}_{collection_name.replace("_part_1", "")}_{embedding_function}_request.csv'
    json_file_path = f'./evaluation/results/{evaluation_file.split("/")[-1].replace(".json", "")}_{collection_name.replace("_part_1", "")}_{embedding_function}_request_raw.json'
    last_index = -1

    # Check if csv file exists
    csv_file_exists = os.path.exists(csv_file_path)
    csv_file = open(csv_file_path, "a")
    csv_writer = None

    # Check if JSON file exists
    json_file_exists = os.path.exists(json_file_path)
    if not json_file_exists:
        json_file = open(json_file_path, "w")
        json_file.write("[]")
    else:
        with open(json_file_path) as f:
            json_file = json.load(f)
            last_index_string = list(json_file[-1].keys())[0]
            # Find a key to attempt to convert to integer (assuming there is at least one key)
            if last_index_string:
                # Convert the last key to integer (will raise ValueError if not possible)
                last_index = int(list(last_index_string)[-1])
                print(f"Starting with index: {last_index}", flush=True)
            else:
                print(
                    "Error: The last dictionary is empty, no keys to convert.",
                    flush=True,
                )
                exit()

    # results = pd.DataFrame(columns=['#Relevant Pages', '#Relevant Pages Retrieved', '#Relevant Documents', '#Relevant Documents Retrieved', '#Relevant Dossiers', '#Relevant Dossiers Retrieved'])
    for index, (key, value) in enumerate(evaluation.items()):
        if index <= last_index:
            print(f"Skipping index {index}", flush=True)
            continue
        results_raw = {}
        if not value.get("pages"):
            print("No pages found in the JSON file")
            continue
        if not value.get("documents"):
            print("No documents found in the JSON file")
            continue
        if not value.get("dossier"):
            print("No dossiers found in the JSON file")
            continue

        # Assuming n == len(value)
        n_pages = len(value["pages"])
        n_documents = len(value["documents"])
        n_dossiers = len(value["dossier"])

        response = querier.ask_question(key)
        print("Asking question of length: ", len(key))
        source_documents = response["source_documents"]

        pages_result = get_first_n_unique_ids_by_type(
            source_documents, n_pages, "page_id"
        )
        documents_result = get_first_n_unique_ids_by_type(
            source_documents, n_documents, "document_id"
        )
        dossiers_result = get_first_n_unique_ids_by_type(
            source_documents, n_dossiers, "dossier_id"
        )

        results_raw[index] = {
            "body": key,
            "pages": pages_result,
            "documents": documents_result,
            "dossier": dossiers_result,
        }

        new_row = [
            len(value["pages"]),  # Relevant Pages
            check_relevance(
                set(value["pages"]), set(pages_result)
            ),  # Relevant Pages Retrieved
            len(value["documents"]),  # Relevant Documents
            check_relevance(
                set(value["documents"]), set(documents_result)
            ),  # Relevant Documents Retrieved
            len(value["dossier"]),  # Relevant Dossiers
            check_relevance(
                set(value["dossier"]), set(dossiers_result)
            ),  # Relevant Dossier Retrieved
        ]

        if csv_writer is None:
            fieldnames = [
                "#Relevant Pages",
                "#Relevant Pages Retrieved",
                "#Relevant Documents",
                "#Relevant Documents Retrieved",
                "#Relevant Dossiers",
                "#Relevant Dossiers Retrieved",
            ]
            csv_writer = pd.DataFrame(columns=fieldnames)
        if not csv_file_exists:
            csv_writer.to_csv(csv_file, index=False, lineterminator="\n")
            csv_file_exists = True  # Prevent header repetition
        csv_writer.loc[len(csv_writer.index)] = new_row
        csv_writer.loc[len(csv_writer.index) - 1 : len(csv_writer.index) - 1].to_csv(
            csv_file, header=False, index=False, lineterminator="\n"
        )
        csv_file.flush()  # Ensure that the data is written to the file in the DHPC environment
        print(f"Index {index} in csv file written.", flush=True)

        # Append or create JSON file
        with open(json_file_path) as json_file:
            all_results_raw = json.load(json_file)
            all_results_raw.append(results_raw)
        with open(json_file_path, "w") as json_file:
            json.dump(all_results_raw, json_file)
        print(f"Index {index} in json file written.", flush=True)


if __name__ == "__main__":
    main()
