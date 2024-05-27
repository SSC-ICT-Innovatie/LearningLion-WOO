import os
import re
import utils as ut
from argparse import ArgumentParser
from langchain_community.vectorstores.chroma import Chroma
import time


def delete_in_batches(target_store, ids_to_remove, batch_size=5461):
    total_items = len(ids_to_remove)
    num_batches = (total_items + batch_size - 1) // batch_size
    print("total_items: ", total_items, flush=True)
    
    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, total_items)
        
        # Get current max id
        batch_ids = ids_to_remove[start_index:end_index]
        try:
            target_store.delete(ids=batch_ids)
        except Exception as e:
            print(f"Error processing batch {i+1}: {e}", flush=True)
        print(f"Succesfully removed batch {i+1}/{num_batches}", flush=True)
    return

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
    
    threshold = 4173911
    # Convert strings to numbers and filter out the ones above the threshold
    ids_to_remove = [num_str for num_str in main_vector_store_data['ids'] if int(num_str) > threshold]
    print("length ids to remove:", len(ids_to_remove), flush=True)
    print("ids to remove:", ids_to_remove, flush=True)
    
    print("Removing ids from the main db...", flush=True)
    # main_vector_store.delete(ids=ids_to_remove)
    delete_in_batches(main_vector_store, ids_to_remove)
    
    print("Done removing ids from the main db.", flush=True)
    print("time taken:", time.time() - start_time, flush=True)
    
    
        
if __name__ == '__main__':
    main()