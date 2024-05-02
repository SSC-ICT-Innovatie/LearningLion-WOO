import pandas as pd
import os
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    # The part number should be replaced with 'NUMBER'
    # For example: WoogleDumps_01-04-2024_12817_dossiers_no_requests_part_NUMBER_chunked
    parser.add_argument('-c', '--content_folder_name', type=str)
    parser.add_argument('-d', '--documents_directory', type=str)
    
    args = parser.parse_args()
    if args.content_folder_name and args.documents_directory:
        content_folder_name = args.content_folder_name
        documents_directory = args.documents_directory
    else:
        print("Please provide the source folder of documents, the output folder name, and the database directory.")
        exit()
    print(f"Source folder of documents: {content_folder_name}")

    # Define the base directory where the CSV files are located and the output directory
    file_name = "woo_merged.csv.gz"
    output_dir = content_folder_name.replace('_part_NUMBER_', '_')
    output_file_name = "woo_merged.csv.gz"

    # Create the output directory if it does not exist
    os.makedirs(os.path.join(documents_directory, output_dir), exist_ok=True)

    # Initialize an empty list to hold dataframes
    dfs = []

    # Loop over the range of directories and read the CSV files
    for i in range(1, 76):
        sub_dir = content_folder_name.replace('NUMBER', str(i))
        file_path = os.path.join(documents_directory, sub_dir, file_name)
        try:
            df = pd.read_csv(file_path, compression='gzip')
        except:
            # Specify dilimiter and quote char, due to presence of special characters, and it might cause an error
            df = pd.read_csv(file_path, compression='gzip', delimiter=',', quotechar='"', engine='python')
        dfs.append(df)

    # Concatenate all dataframes into one
    final_df = pd.concat(dfs, ignore_index=True)

    # Save the concatenated dataframe to a compressed CSV
    output_path = os.path.join(documents_directory, output_dir, output_file_name)
    final_df.to_csv(output_path, index=False, compression='gzip')

if __name__ == "__main__":
    main()