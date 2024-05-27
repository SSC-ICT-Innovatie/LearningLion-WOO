# Merge vector dbs to create a single vector db

import os
import re
import utils as ut
from argparse import ArgumentParser
from langchain_community.vectorstores.chroma import Chroma
import time

def merge_in_batches(target_store, source_data, max_id, batch_size=5461):
    total_items = len(source_data['embeddings'])
    num_batches = (total_items + batch_size - 1) // batch_size
    print("total_items: ", total_items, flush=True)
    
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, total_items)
        
        # Get current max id
        batch_ids = [str(x) for x in range(max_id + 1, max_id + end_index - start_index + 1)]
        batch_embeddings = source_data['embeddings'][start_index:end_index]
        batch_metadatas = source_data['metadatas'][start_index:end_index]
        batch_documents = source_data['documents'][start_index:end_index]
        try:
            target_store._collection.add(
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                documents=batch_documents,
                ids=batch_ids
            )
            
        except Exception as e:
            print(f"Error processing batch {i+1}: {e}", flush=True)
        print(f"Succesfully added batch {i+1}/{num_batches}", flush=True)
        max_id = int(batch_ids[-1])
    return max_id
        
# Define a function to extract the part number from each path
def extract_part_number(path):
    # This pattern matches 'part_xx' where xx is the part number
    match = re.search(r'part_(\d+)', path)
    if match:
        return int(match.group(1))
    return 0 # Return 0 or some default value if no part number is found

def main():
    # Parse all the arguments and read the settings
    parser = ArgumentParser()
    parser.add_argument('-b', '--base_collection_name', type=str)
    parser.add_argument('-c', '--merge_collection_name', type=str)
    parser.add_argument('-p', '--embedding_provider', type=str)
    parser.add_argument('-a', '--embedding_author', type=str)
    parser.add_argument('-f', '--embedding_function', type=str)
    parser.add_argument('-v', '--base_vector_db_folder', type=str)
    parser.add_argument('-m', '--merge_vector_db_directory', type=str)
    
    args = parser.parse_args()
    if args.base_collection_name and args.merge_collection_name and args.embedding_provider and args.embedding_author and args.embedding_function and args.base_vector_db_folder and args.merge_vector_db_directory:
        base_collection_name = args.base_collection_name
        merge_collection_name = args.merge_collection_name
        embedding_provider = args.embedding_provider
        embedding_author = args.embedding_author
        embedding_function = args.embedding_function
        complete_embedding_function = f"{embedding_author}/{embedding_function}"
        base_vector_db_folder = args.base_vector_db_folder
        merge_vector_db_directory = args.merge_vector_db_directory
    else:
        pass
        print("Please provide all the necessary arguments.", flush=True)
        exit()
        
    # base_collection_name = "WoogleDumps_01-04-2024_12817_dossiers_no_requests_part_1"
    # # Replace the number with "NUMBER", so that it can get all of the collection names
    # merge_collection_name = "WoogleDumps_01-04-2024_12817_dossiers_no_requests_part_NUMBER"
    # embedding_provider = "local_embeddings"
    # embedding_author = "GroNLP"
    # embedding_function = "bert-base-dutch-cased"
    # complete_embedding_function = f"{embedding_author}/{embedding_function}"
    # base_vector_db_folder = f"./vector_stores/no_requests_all_parts_chromadb_1024_256_local_embeddings_GroNLP/{embedding_function}"
    # merge_vector_db_directory = "./vector_stores/WoogleDumps_01-04-2024_12817_dossiers_no_requests_chromadb_1024_256_local_embeddings_GroNLP_parts"
    
    # List all entries in the directory
    entries = os.listdir(merge_vector_db_directory)

    # Get all the folders to add
    folders = [os.path.join(merge_vector_db_directory, entry, embedding_function) for entry in entries if os.path.isdir(os.path.join(merge_vector_db_directory, entry))]
    sorted_folders = sorted(folders, key=extract_part_number)
    
    # Get all the base_collection_names, and put them in a tuple with the folders to add
    # Let the index start with 2, because part 1 is considered the base_collection_name
    folder_tuple = [(merge_collection_name.replace('NUMBER', str(index + 2)), value) for index, value in enumerate(sorted_folders)]
    
    embeddings = ut.getEmbeddings(embedding_provider, complete_embedding_function, None, None)
    
    start_time = time.time()
    
    print("Reading the main db...", flush=True)    
    main_vector_store = Chroma(
        collection_name=base_collection_name,
        embedding_function=embeddings,
        persist_directory=base_vector_db_folder,
        collection_metadata={"hnsw:space": "cosine"}
    )
    main_vector_store_data = main_vector_store.get()    
    print("length main db:", len(main_vector_store_data['ids']), flush=True)
    max_id = max([int(num) for num in main_vector_store_data['ids']])
    print(f"Time it took to read the main db: {time.time() - start_time}", flush=True)
    
    for index, (collection_name, vector_db_folder) in enumerate(folder_tuple):
        if index < 29:
            continue
        tuple_time = time.time()
        print("Currently processing: ", collection_name, flush=True)
        # We are rereading the main db every time, due to an issue of the db ids not being properly updated.

        print("Reading the db to add...", flush=True)
        vector_store_to_add = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=vector_db_folder,
            collection_metadata={"hnsw:space": "cosine"}
        )
        vector_store_to_add_data = vector_store_to_add.get(include=['documents','metadatas','embeddings'])
        
        # Generate the new ids
        print("Main db max id: ", max_id, flush=True)
        max_id = merge_in_batches(main_vector_store, vector_store_to_add_data, max_id)
        print("Max id after merge: ", max_id, flush=True)
        print(f"Time it took to process {collection_name}: {time.time() - tuple_time}", flush=True)
        print(f"Total time it took to process {collection_name}: {time.time() - start_time}", flush=True)
        print("=======================================")
        
    
if __name__ == "__main__":
    main()