"""
Example with arguments:
python preprocess_whole_document.py --content_folder_name 12_dossiers_no_requests --documents_directory docs
"""

import os
import pandas as pd
from argparse import ArgumentParser

def main():
    parser = ArgumentParser()
    parser.add_argument("--content_folder_name", type=str, required=True)
    parser.add_argument("--documents_directory", type=str, required=True)
    args = parser.parse_args()

    # Selecting the paths
    input_path = os.path.join(args.documents_directory, args.content_folder_name, "woo_merged.csv.gz")
    woo_data = pd.read_csv(input_path, compression="gzip")

    woo_data = (
        woo_data.groupby("document_id")
        .agg(
            {
                "bodyText": lambda x: " ".join(x.astype(str)).replace('\n', ' '),
                "page_id": "first",
                "dossier_id": "first",
            }
        )
        .reset_index()
    )

    new_file_name = args.content_folder_name + "_whole_document"
    save_directory = os.path.join(args.documents_directory, new_file_name)
    if os.path.exists(save_directory):
        raise FileExistsError(f"[Error] ~ The directory '{save_directory}' already exists.", flush=True)
    os.makedirs(save_directory)
    output_path = os.path.join(save_directory, "woo_merged.csv.gz")
    woo_data.to_csv(output_path, compression="gzip", index=False)
    print(f"[Info] ~ Processed data saved to {output_path}.")

if __name__ == "__main__":
    main()
