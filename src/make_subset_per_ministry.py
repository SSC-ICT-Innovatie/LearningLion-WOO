"""
This script will make 12 subsets of woo-dossiers. Every subset will contain dossiers of a different ministry
It will generate a subset without the requests and with the requests. It will also only keep the necessary columns and rows.

Example with arguments:
python preprocess_make_subset_per_ministry.py --content_folder_name WoogleDumps_01-04-2024_12817_dossiers --documents_directory ./docs --save_directory ./docs_ministries_full
"""

import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser


def keep_relevant_columns(df):
    # Explicitly create a copy of the DataFrame to avoid setting-with-copy warning
    df = df.copy()

    # Use .loc to ensure that changes are made directly in the copied DataFrame
    df.loc[:, "all_foi_bodyText"] = np.where(
        df["bodytext_foi_bodyTextOCR"].notnull() & df["bodytext_foi_bodyTextOCR"].str.strip().ne(""),
        df["bodytext_foi_bodyTextOCR"],
        df["bodytext_foi_bodyText"],
    )

    # Remove newline characters from the bodyText column
    df.loc[:, "all_foi_bodyText"] = df["all_foi_bodyText"].str.replace("\n", " ", regex=False).str.replace("  ", " ", regex=False)

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


def keep_relevant_rows(df):
    # Explicitly create a copy of the DataFrame to avoid setting-with-copy warning
    df = df.copy()

    # Filter rows where 'bodyText' is of type str and not empty
    return df[df["bodyText"].apply(lambda x: isinstance(x, str) and bool(x.strip()))]


def get_first_n_dossiers_by_ministry(tuples, n, ministry, closest_nr_docs=20):
    # Filter tuples by the specified ministry and calculate the difference from 20
    filtered_and_diff = [(dossier_id, abs(num - closest_nr_docs)) for dossier_id, dossier_ministry, num in tuples if dossier_ministry.lower() == ministry.lower()]

    # Sort by the difference (second element in the tuple)
    sorted_by_closest = sorted(filtered_and_diff, key=lambda x: x[1])

    # Get the top n dossiers
    results = [dossier_id for dossier_id, diff in sorted_by_closest[:n]]

    if len(results) < n:
        print(f"[Warning] ~ Only {len(results)} dossiers found for {ministry}, with n={n}.")

    return results


def make_unique_tuple(tuple_list):
    # Dictionary to store ID as key and a tuple of (publisher, count) as value
    count_dict = {}

    # Iterate over each tuple in the input list
    for id, publisher in tuple_list:
        if id in count_dict:
            # Increase the count for this ID
            current_publisher, current_count = count_dict[id]
            count_dict[id] = (current_publisher, current_count + 1)
        else:
            # Add new ID with the publisher and initial count of 1
            count_dict[id] = (publisher, 1)

    # Generate the list of unique tuples with counts
    return [(id, publisher, count) for id, (publisher, count) in count_dict.items()]


def main():
    parser = ArgumentParser()
    parser.add_argument("--content_folder_name", type=str, required=True)
    parser.add_argument("--documents_directory", type=str, required=True)
    parser.add_argument("--save_directory", type=str, required=True)
    args = parser.parse_args()

    content_folder_name = args.content_folder_name
    documents_directory = args.documents_directory
    save_directory = args.save_directory

    print(f"Source folder of documents: {content_folder_name}")

    file_path = f"{documents_directory}/{content_folder_name}/woo_merged.csv.gz"
    woo_data = pd.read_csv(file_path, compression="gzip")

    filtered_df = woo_data[woo_data["dossiers_dc_publisher_name"].str.contains("Ministerie", case=False)]

    # Get all dossiers that are from ministries and have a request or decision file
    filtered_requests = woo_data[
        (woo_data["dossiers_dc_publisher_name"].str.contains("Ministerie", case=False)) & ((woo_data["documents_dc_type"] == "verzoek") | (woo_data["documents_dc_type"] == "besluit"))
    ]

    # Filter for dossiers that have an attachment ("bijlage")
    filtered_attachments = woo_data[(woo_data["dossiers_dc_publisher_name"].str.contains("Ministerie", case=False)) & (woo_data["documents_dc_type"] == "bijlage")]

    # Get unique dossier IDs from both filters
    dossier_ids_requests = set(filtered_requests["foi_dossierId"])
    dossier_ids_attachments = set(filtered_attachments["foi_dossierId"])

    # Find intersection of dossier IDs that have either "verzoek" or "besluit" and also have at least one "bijlage"
    dossier_ids_combined = dossier_ids_requests.intersection(dossier_ids_attachments)

    # Filter the original data to get rows that match the combined dossier IDs
    final_filtered_df = woo_data[woo_data["foi_dossierId"].isin(dossier_ids_combined)]

    # Create tuples of dossier IDs and publisher names from the final filtered DataFrame
    dossier_ids_with_publishers = list(zip(final_filtered_df["foi_dossierId"], final_filtered_df["dossiers_dc_publisher_name"]))

    final_dossier_tuples = make_unique_tuple(dossier_ids_with_publishers)

    ministries = [
        "ministerie van landbouw, natuur en voedselkwaliteit",
        "ministerie van financiën",
        "ministerie van justitie en veiligheid",
        "ministerie van infrastructuur en waterstaat",
        "ministerie van economische zaken en klimaat",
        "ministerie van binnenlandse zaken en koninkrijksrelaties",
        "ministerie van sociale zaken en werkgelegenheid",
        "ministerie van volksgezondheid, welzijn en sport",
        "ministerie van buitenlandse zaken",
        "ministerie van onderwijs, cultuur en wetenschap",
        "ministerie van defensie",
        "ministerie van algemene zaken",
    ]

    for ministry in ministries:
        dossiers = get_first_n_dossiers_by_ministry(final_dossier_tuples, 999, ministry, 1)

        # Filter the DataFrame for the dossiers
        filtered_df_with_requests = filtered_df[filtered_df["foi_dossierId"].isin(dossiers)]
        filtered_df_no_requests = filtered_df_with_requests[((filtered_df_with_requests["documents_dc_type"] != "verzoek") & (filtered_df_with_requests["documents_dc_type"] != "besluit"))]

        relevant_data = keep_relevant_columns(filtered_df_with_requests)
        relevant_data_no_requests = keep_relevant_columns(filtered_df_no_requests)

        filtered_relevant_data = keep_relevant_rows(relevant_data)
        filtered_relevant_data_no_requests = keep_relevant_rows(relevant_data_no_requests)

        num_dossiers = filtered_relevant_data["dossier_id"].nunique()
        num_dossiers_no_requests = filtered_relevant_data_no_requests["dossier_id"].nunique()

        ministry_folder = f"{save_directory}/{ministry.replace(' ', '_').replace(',', '')}".lower()
        ministry_folder_no_requests = f"{save_directory}/{ministry.replace(' ', '_').replace(',', '')}_no_requests".lower()

        os.makedirs(ministry_folder, exist_ok=True)
        os.makedirs(ministry_folder_no_requests, exist_ok=True)

        filtered_relevant_data.to_csv(
            os.path.join(ministry_folder, "woo_merged.csv.gz"),
            index=False,
            compression="gzip",
        )
        filtered_relevant_data_no_requests.to_csv(
            os.path.join(ministry_folder_no_requests, "woo_merged.csv.gz"),
            index=False,
            compression="gzip",
        )
        print(
            f"[Info] ~ Subset for {ministry} created with {num_dossiers} dossiers and {num_dossiers_no_requests} without requests.",
            flush=True,
        )


if __name__ == "__main__":
    main()
