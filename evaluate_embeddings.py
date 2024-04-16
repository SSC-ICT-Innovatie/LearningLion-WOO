import json
import os
import pandas as pd
import settings
import utils as ut
from query.querier import Querier
from truncate_merge_woo import select_woogle_dump_folders

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

def get_first_n_unique_ids_by_type(source_documents: list, n: int, id_type: str) -> list:
    """
    Extracts the first n unique document IDs from a list of source documents.
    
    Parameters:
    - source_documents: A list of tuples, where each tuple contains a document object and another value.
    - n: The number of unique document IDs to retrieve.
    
    Returns:
    A list of the first n unique document IDs.
    """
    
    if id_type not in ["id", "foi_documentId", "foi_dossierId"]:
        raise ValueError("id_type must be 'id', 'foi_documentId', or 'type2'")
    
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

# Selecting the paths
path = select_woogle_dump_folders(path='../docs')
evaluation_file = "evaluation_request_WoogleDumps_01-04-2024_50_dossiers_no_requests.json"

querier = Querier()
content_folder_name, vectordb_folder_path = ut.create_vectordb_name(path.split('/')[-1])
vectordb_folder_path = "../" + vectordb_folder_path
content_folder_name = content_folder_name.split('/')[-1].split('\\')[-1]

# If vector store folder does not exist, stop
if not os.path.exists(vectordb_folder_path):
    print("There is no vector database for this folder yet. First run \"python ingest.py\"")
    ut.exit_program()

querier.make_chain(content_folder_name, vectordb_folder_path)

with open(f"../evaluation/{evaluation_file}", 'r') as file:
    json_data = json.load(file)
    for i in range(2):
        results_raw = {}
        results = pd.DataFrame(columns=['#Relevant Pages', '#Relevant Pages Retrieved', '#Relevant Documents', '#Relevant Documents Retrieved', '#Relevant Dossiers', '#Relevant Dossiers Retrieved'])
        for key, value in json_data.items():
            if not value.get('pages'):
                print("No pages found in the JSON file")
                continue
            if not value.get('documents'):
                print("No documents found in the JSON file")
                continue
            if not value.get('dossier'):
                print("No dossiers found in the JSON file")
                continue

            if i == 0:
                n_pages = len(value['pages'])
                n_documents = len(value['documents'])
                n_dossiers = len(value['dossier'])
            else:
                n_pages = 1
                n_documents = 1
                n_dossiers = 1
        
            response = querier.ask_question(key)
            source_documents = response["source_documents"]
            
            pages_result = get_first_n_unique_ids_by_type(source_documents, n_pages, "id")
            documents_result = get_first_n_unique_ids_by_type(source_documents, n_documents, "foi_documentId")
            dossiers_result = get_first_n_unique_ids_by_type(source_documents, n_dossiers, "foi_dossierId")
            
            results_raw[key] = {
                'pages': pages_result,
                'documents': documents_result,
                'dossier': dossiers_result
            }
            
            new_row = [
                len(value['pages']), #Relevant Pages
                check_relevance(set(value['pages']), set(pages_result)), #Relevant Pages Retrieved
                len(value['documents']), #Relevant Documents
                check_relevance(set(value['documents']), set(documents_result)), #Relevant Documents Retrieved
                len(value['dossier']), #Relevant Dossiers
                check_relevance(set(value['dossier']), set(dossiers_result)), #Relevant Dossier Retrieved
            ]
            results.loc[len(results.index)] = new_row
        base_directory = '../evaluation/results'
        csv_file_name = f'{evaluation_file.split(".")[0]}_{"1" if i == 1 else "n"}_{settings.EMBEDDINGS_MODEL.split("/")[-1]}_request.csv'
        csv_file_path = os.path.join(base_directory, csv_file_name)

        # Check if the /results directory exists, if not, create it
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)

        results.to_csv(csv_file_path, index=False)

        # Saving results_raw dictionary to a JSON file
        json_file_name = f'{evaluation_file.split(".")[0]}_{"1" if i == 1 else "n"}_{settings.EMBEDDINGS_MODEL.split("/")[-1]}_request_raw.json'
        json_file_path = os.path.join(base_directory, json_file_name)

        with open(json_file_path, 'w') as file:
            json.dump(results_raw, file)
