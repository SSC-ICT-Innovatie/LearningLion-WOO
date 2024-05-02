# Splits the woo_merged csv file into smaller parts of 25,000 rows each.
# Only keeps the necessary 4 columns: 'pageId', 'documentId', 'dossierId', 'bodyText'.
# This is necessary to avoid memory issues and boost computing time when
# processing the data in future calculations.

import numpy as np
import os
import pandas as pd
import settings
from argparse import ArgumentParser

def main():
    # Parse all the arguments and read the settings
    parser = ArgumentParser(description="Document ingestion script using the Ingester class.")
    parser.add_argument('-c', '--content_folder_name', type=str)
    parser.add_argument('-d', '--documents_directory', type=str)
    parser.add_argument('-s', '--csv_size', type=int)
    
    args = parser.parse_args()
    if args.content_folder_name:
        content_folder_name = args.content_folder_name
    else:
        print("Source folder not specified.")
        exit()
    print(f"Source folder of documents: {content_folder_name}")
    documents_directory = args.documents_directory if args.documents_directory else settings.DOC_DIR
    csv_size = args.csv_size if args.csv_size else 25000

    file_path = f'{documents_directory}/{content_folder_name}/woo_merged.csv.gz'
    woo_data = pd.read_csv(file_path, compression='gzip')
    woo_data.reset_index(inplace=True)
    
    # First take bodyTextOCR if it is not empty, else take bodyText
    woo_data['all_foi_bodyText'] = np.where(woo_data['bodytext_foi_bodyTextOCR'].notnull() & woo_data['bodytext_foi_bodyTextOCR'].str.strip().ne(''), woo_data['bodytext_foi_bodyTextOCR'], woo_data['bodytext_foi_bodyText'])

    # Only keep the necessary columns, and rename them
    rename_dict = {
        'id': 'page_id',
        'foi_documentId': 'document_id',
        'foi_dossierId': 'dossier_id',
        'all_foi_bodyText': 'bodyText'
    }

    # New DataFrame with renamed and subset of columns
    new_woo_data = woo_data.rename(columns=rename_dict)[['page_id', 'document_id', 'dossier_id', 'bodyText']]
    
    number_of_chunks = (len(new_woo_data) + csv_size - 1) // csv_size  # Ceiling division to handle any remainder
    # Split the DataFrame into chunks of csv_size rows and write each to a new compressed CSV file
    for i in range(number_of_chunks):
        # Define the start and end of each chunk
        start_row = i * csv_size
        end_row = start_row + csv_size
        chunk_df = new_woo_data.iloc[start_row:end_row]

        # Define and create a directory for this chunk
        directory_path = f'{documents_directory}/{content_folder_name}_part_{i+1}'
        os.makedirs(directory_path, exist_ok=True)  # Create the directory if it does not exist

        # Write the chunk to a new compressed CSV file
        chunk_file_path = os.path.join(directory_path, 'woo_merged.csv.gz')
        chunk_df.to_csv(chunk_file_path, index=False, compression='gzip')

if __name__ == "__main__":
    main()