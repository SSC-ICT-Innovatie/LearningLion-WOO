"""
Creates a vector store given a WOO dataset and embeddings model.

Example with arguments:
python ingest.py --embeddings_model GroNLP/bert-base-dutch-cased --content_folder_name minaz_no_requests --documents_directory ./docs_ministries_full --vector_store_folder ./vector_stores
python ingest.py --embeddings_model meta-llama/Meta-Llama-3-8B --content_folder_name minaz_no_requests --documents_directory ./docs --vector_store_folder ./vector_stores
"""

import os
import pandas as pd
from argparse import ArgumentParser
from common import chroma
from common import embeddings as emb
from common.ingestutils import IngestUtils
from common.register_time import Timer
from dotenv import load_dotenv
from typing import Dict, List, Tuple

# Local settings
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256
TEXT_SPLITTER_METHOD = "NLTKTextSplitter"
VECDB_TYPE = "chromadb"


def create_vectordb_name(content_folder_name: str, embeddings_model: str, documents_directory: str, vector_store_folder: str) -> Tuple[str, str]:
    """
    Constructs the paths for the content folder and vector database folder based on input parameters.

    Parameters:
        content_folder_name (str): The name of the content folder.
        embeddings_model (str): The model name used for embeddings.
        documents_directory (str): The base directory where documents are stored.
        vector_store_folder (str): The base directory for storing vector databases.

    Returns:
        Tuple[str, str]: A tuple containing the path to the content folder and the path to the vector database folder.
    """
    content_folder_path = os.path.join(documents_directory, content_folder_name)
    vectordb_name = f"{content_folder_name}_{VECDB_TYPE}_{CHUNK_SIZE}_{CHUNK_OVERLAP}_{embeddings_model}"
    vectordb_folder_path = os.path.join(vector_store_folder, vectordb_name)
    return content_folder_path, vectordb_folder_path


def parse_woo(woo: pd.core.series.Series) -> Tuple[List[Tuple[int, str]], Dict[str, any]]:
    """
    Parses a pandas Series to extract text from 'bodyText' and separate the remaining data as metadata.

    Parameters:
        woo (pd.core.series.Series): A Series containing a 'bodyText' key among other data.

    Returns:
        Tuple[List[Tuple[int, str]], Dict[str, any]]: A tuple where the first element is a list
        containing a single tuple with the index and text content, and the second element is a
        dictionary of the remaining series data as metadata.
    """

    woo_json = woo.to_dict()

    # If type is not string, return None
    if type(woo_json["bodyText"]) != str:
        return None, None

    # Hardcode the array of tuple, because the way it is currently coded there is only 1 tuple in every page
    tuple_bodyText = [(0, woo_json["bodyText"])]

    # Delete the 'bodyText' key from the dictionary, because it should not be included in the metadata
    woo_json.pop("bodyText", None)

    return tuple_bodyText, woo_json


def ingest(content_folder_name, content_folder_path, vectordb_folder_path, embeddings_model, timer):
    load_dotenv()
    ingestutils = IngestUtils(CHUNK_SIZE, CHUNK_OVERLAP, TEXT_SPLITTER_METHOD)

    # Get embeddings and read data
    embeddings = emb.getEmbeddings(embeddings_model)
    woo_data_path = os.path.join(content_folder_path, "woo_merged.csv.gz")
    woo_data = pd.read_csv(woo_data_path)

    if VECDB_TYPE == "chromadb":
        vector_store = chroma.get_chroma_vector_store(content_folder_name, embeddings, vectordb_folder_path)
        collection = vector_store.get()
        collection_ids = [int(id) for id in collection["ids"]]
        files_in_store = list(set(metadata["document_id"] for metadata in collection["metadatas"]))

        # Check if there are any deleted items
        files_deleted = [file for file in files_in_store if file not in woo_data["document_id"].tolist()]
        if len(files_deleted) > 0:
            print(f"[Info] ~ Files are deleted, so vector store for {content_folder_name} needs to be updated.", flush=True)
            idx_id_to_delete = []
            for idx in range(len(collection["ids"])):
                idx_id = collection["ids"][idx]
                idx_metadata = collection["metadatas"][idx]
                if idx_metadata["document_id"] in files_deleted:
                    idx_id_to_delete.append(idx_id)
            vector_store.delete(idx_id_to_delete)
            print(f"[Info] ~ {len(collection['ids'])} files deleted from vectorstore.", flush=True)

        # Check if there is new data and only keep the new data
        woo_data = woo_data[~woo_data["document_id"].isin(files_in_store)]
        collection = vector_store.get()
        collection_ids = [int(id) for id in collection["ids"]]
        if len(collection_ids) == 0:
            start_id = 0
        else:
            start_id = max(collection_ids) + 1
        if len(woo_data) > 0:
            print(f"[Info] ~ Files are added, so vector store for {content_folder_name} needs to be updated", flush=True)
            for index, row in woo_data.reset_index().iterrows():
                # Extract raw text pages and metadata
                raw_pages, metadata = parse_woo(row)
                if raw_pages is None or metadata is None:
                    continue

                # Convert the raw text to cleaned text chunks
                documents = ingestutils.clean_text_to_docs(raw_pages, metadata)

                # If there are no documents, continue to the next iteration
                if len(documents) == 0:
                    continue

                try:
                    ids = [str(id) for id in list(range(start_id, start_id + len(documents)))]
                    print(f"[Info] ~ Now processing {row.get('page_id')}, with id's: {', '.join(map(str,ids))}.", flush=True)
                    vector_store.add_documents(documents=documents, embedding=embeddings, collection_name=content_folder_name, persist_directory=vectordb_folder_path, ids=ids)
                    collection = vector_store.get()
                    collection_ids = [int(id) for id in collection["ids"]]
                    start_id = max(collection_ids) + 1
                except Exception as e:
                    print(f"[Error] ~ Error adding documents to vector store: {e}", flush=True)
                    continue
            # Save updated vector store to disk
            vector_store.persist()
            timer.update_time()
        else:
            print("[Warning] ~ No new woo documents to be ingested", flush=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--embeddings_model", type=str, required=True)
    parser.add_argument("--content_folder_name", type=str, required=True)
    parser.add_argument("--documents_directory", type=str, required=True)
    parser.add_argument("--vector_store_folder", type=str, required=True)
    args = parser.parse_args()

    content_folder_path, vectordb_folder_path = create_vectordb_name(args.content_folder_name, args.embeddings_model, args.documents_directory, args.vector_store_folder)
    
    timer = Timer(args.content_folder_name, args.embeddings_model, ingest=True, folder_name=args.documents_directory)

    ingest(args.content_folder_name, content_folder_path, vectordb_folder_path, args.embeddings_model, timer)


if __name__ == "__main__":
    main()
