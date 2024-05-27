import numpy as np
import os
import pandas as pd
import settings
from argparse import ArgumentParser
from ingest.ingest_utils import IngestUtils
from ingest.woo_parser import WooParser

def main():
    parser = ArgumentParser()
    parser.add_argument('-c', '--content_folder_name', type=str)
    parser.add_argument('-d', '--documents_directory', type=str)
    
    args = parser.parse_args()
    # Get source folder with docs from user
    if args.content_folder_name:
        content_folder_name = args.content_folder_name
    else:
        print("Please provide the source folder of documents.")
        exit()
    print(f"Source folder of documents: {content_folder_name}")
    documents_directory = args.documents_directory if args.documents_directory else settings.DOC_DIR

    # Read the merged woo file
    ingestutils = IngestUtils(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP, None, settings.TEXT_SPLITTER_METHOD)
    woo_data = pd.read_csv(f'{documents_directory}/{content_folder_name}/woo_merged.csv.gz')
    woo_data.reset_index(inplace=True)
    
    woo_parser = WooParser()
    results = pd.DataFrame(columns=['chunk_id', 'page_id', 'document_id', 'dossier_id', 'bodyText'])

    for index, row in woo_data.iterrows():
        raw_pages, metadata = woo_parser.parse_woo_simple(row)
        if raw_pages is None or metadata is None:
            continue
        print("raw_pages: ", raw_pages)
        
        # Convert the raw text to cleaned text chunks
        documents = ingestutils.clean_text_to_docs(raw_pages, metadata)
        print("documents: ", documents)
        
        # If there are no documents, continue to the next iteration
        if len(documents) == 0:
            continue
        
        for document in documents:
            print("document: ", document.page_content, document.metadata['page_id'])
            if document.page_content is None or document.metadata is None:
                continue
            
            new_row = [
                f"{document.metadata['page_id']}.chunk.{document.metadata['chunk']}",
                document.metadata['document_id'],
                document.metadata['dossier_id'],
                document.metadata['page_id'],
                document.page_content
            ]
            results.loc[len(results.index)] = new_row
            
    save_folder = content_folder_name + "_chunked"
        
    # Save results to CSV file
    # Create the directory if it does not exist
    output_dir = os.path.join(documents_directory, save_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Save results to CSV file
    results.to_csv(f'{output_dir}/woo_merged.csv.gz', index=False, compression='gzip')
    
if __name__ == "__main__":
    main()