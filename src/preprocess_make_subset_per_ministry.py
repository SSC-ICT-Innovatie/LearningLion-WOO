# This script will make 12 subsets of woo-dossiers. Every subset will contain dossiers of a different ministry
# It will generate a subset without the requests and with the requests It will also only keep the necessary columns.

# Example with arguments:
# python preprocess_make_subset_per_ministry.py --content_folder_name WoogleDumps_01-04-2024_12817_dossiers --documents_directory ./docs --save_directory ./docs

import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser


def keep_relevant_columns(df):
    # Explicitly create a copy of the DataFrame to avoid setting-with-copy warning
    df = df.copy()

    # Use .loc to ensure that changes are made directly in the copied DataFrame
    df.loc[:, "all_foi_bodyText"] = np.where(
        df["bodytext_foi_bodyTextOCR"].notnull()
        & df["bodytext_foi_bodyTextOCR"].str.strip().ne(""),
        df["bodytext_foi_bodyTextOCR"],
        df["bodytext_foi_bodyText"],
    )

    # Define the columns to rename
    rename_dict = {
        "id": "page_id",
        "foi_documentId": "document_id",
        "foi_dossierId": "dossier_id",
        "all_foi_bodyText": "bodyText",
        "documents_dc_type": "type",
        "dossiers_dc_publisher_name": "publisher",
        "documents_dc_source": "source",
    }

    # Rename the columns
    df = df.rename(columns=rename_dict)

    # Select and return only the necessary columns
    return df[
        [
            "page_id",
            "document_id",
            "dossier_id",
            "bodyText",
            "type",
            "publisher",
            "source",
        ]
    ]


def main():
    parser = ArgumentParser()
    parser.add_argument("--content_folder_name", type=str, required=True)
    parser.add_argument("--documents_directory", type=str, required=True)
    parser.add_argument("--save_directory", type=str, required=True)
    args = parser.parse_args()

    content_folder_name = args.content_folder_name
    documents_directory = args.documents_directory
    save_directory = args.save_directory

    file_path = os.path.join(documents_directory, content_folder_name, "woo_merged.csv.gz")
    woo_data = pd.read_csv(file_path, compression="gzip")

    ministries = woo_data['dossiers_dc_publisher_name'].unique()

    for ministry in ministries:
        if "Ministerie" in ministry:
            filtered_df = woo_data[woo_data['dossiers_dc_publisher_name'] == ministry]
            filtered_df_no_requests = filtered_df[filtered_df['documents_dc_type'] != "verzoek"]

            relevant_data = keep_relevant_columns(filtered_df)
            relevant_data_no_requests = keep_relevant_columns(filtered_df_no_requests)

            num_dossiers = relevant_data['dossier_id'].nunique()
            num_dossiers_no_requests = relevant_data_no_requests['dossier_id'].nunique()

            ministry_folder = f"{save_directory}/{ministry.replace(' ', '_').replace(',', '')}".lower()
            ministry_folder_no_requests = f"{save_directory}/{ministry.replace(' ', '_').replace(',', '')}_no_requests".lower()
            
            os.makedirs(ministry_folder, exist_ok=True)
            os.makedirs(ministry_folder_no_requests, exist_ok=True)

            relevant_data.to_csv(
                os.path.join(ministry_folder, "woo_merged.csv.gz"),
                index=False, compression="gzip"
            )
            relevant_data_no_requests.to_csv(
                os.path.join(ministry_folder_no_requests, "woo_merged.csv.gz"),
                index=False, compression="gzip"
            )
            print(f"[Info] ~ Subset for {ministry} created with {num_dossiers} dossiers and {num_dossiers_no_requests} without requests.", flush=True)



if __name__ == "__main__":
    main()
