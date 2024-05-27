# This script will make a subset of the most recent woo-dossiers, of the 12 different ministries.
# It will take the 1, 5 of each ministry that have a valid request file,
# resulting in a total of 12, 60 dossiers respectively. It will also only keep the necessary columns.
# It will also create a complete subset of all dossiers, with the necessary columns.

# Example with arguments:
# python preprocess_make_subset_ministry.py -c WoogleDumps_01-04-2024_12817_dossiers -d ./docs -s ./docs

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


def get_first_n_dossiers_by_ministry(tuples, n, ministry, closest_nr_docs=20):
    # Filter tuples by the specified ministry and calculate the difference from 20
    filtered_and_diff = [
        (dossier_id, abs(num - closest_nr_docs))
        for dossier_id, dossier_ministry, num in tuples
        if dossier_ministry.lower() == ministry.lower()
    ]

    # Sort by the difference (second element in the tuple)
    sorted_by_closest = sorted(filtered_and_diff, key=lambda x: x[1])

    # Get the top n dossiers
    results = [dossier_id for dossier_id, diff in sorted_by_closest[:n]]

    if len(results) < n:
        print(
            f"[Warning] ~ Only {len(results)} dossiers found for {ministry}, with n={n}."
        )

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


def intersection_of_tuples(list1, list2):
    # Create a dictionary from list1 where key is the first element of each tuple
    dict1 = {t[0]: t for t in list1}

    # Create a set from the first elements of each tuple in list2
    set2 = {t[0] for t in list2}

    # Find intersection of keys from dict1 and elements in set2
    intersection_keys = set(dict1.keys()) & set2

    # Retrieve the full tuples from dict1 based on the intersection keys
    return [dict1[key] for key in intersection_keys]


def main():
    parser = ArgumentParser(
        description="Document ingestion script using the Ingester class."
    )
    parser.add_argument("-c", "--content_folder_name", type=str)
    parser.add_argument("-d", "--documents_directory", type=str)
    parser.add_argument("-s", "--save_directory", type=str)
    args = parser.parse_args()

    if args.content_folder_name and args.documents_directory:
        content_folder_name = args.content_folder_name
        documents_directory = args.documents_directory
        save_directory = args.save_directory
    else:
        print("Not all arguments are provided")
        exit()
    print(f"Source folder of documents: {content_folder_name}")

    file_path = f"{documents_directory}/{content_folder_name}/woo_merged.csv.gz"
    woo_data = pd.read_csv(file_path, compression="gzip")

    filtered_df = woo_data[
        woo_data["dossiers_dc_publisher_name"].str.contains("Ministerie", case=False)
    ]
    # Get all dossiers that are from ministries and have a request file
    filtered_df_request = woo_data[
        (woo_data["dossiers_dc_publisher_name"].str.contains("Ministerie", case=False))
        & (woo_data["documents_dc_type"] == "verzoek")
    ]
    dossier_ids_with_publishers_request = list(
        zip(
            filtered_df_request["foi_dossierId"],
            filtered_df_request["dossiers_dc_publisher_name"],
        )
    )
    dossier_tuple_request = make_unique_tuple(dossier_ids_with_publishers_request)

    # Get all dossiers that are from ministries and have an attachment file
    filtered_df_attachment = woo_data[
        (woo_data["dossiers_dc_publisher_name"].str.contains("Ministerie", case=False))
        & (woo_data["documents_dc_type"] == "bijlage")
    ]
    dossier_ids_with_publishers_attachment = list(
        zip(
            filtered_df_attachment["foi_dossierId"],
            filtered_df_attachment["dossiers_dc_publisher_name"],
        )
    )
    dossier_tuple_attachment = make_unique_tuple(dossier_ids_with_publishers_attachment)

    # Make a full tuple of all dossiers
    full_dossier_ids = list(
        zip(
            filtered_df_attachment["foi_dossierId"],
            filtered_df_attachment["dossiers_dc_publisher_name"],
        )
    )
    full_tuple = make_unique_tuple(full_dossier_ids)

    # Get the intersection of the two sets, with the number of attachments as quantity number
    unique_foi_dossier_ids = intersection_of_tuples(
        dossier_tuple_attachment, dossier_tuple_request
    )

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

    number_of_dossiers = [1, 5]
    for i in number_of_dossiers:
        result = pd.DataFrame()
        result_no_requests = pd.DataFrame()
        nr_of_dossiers = 0
        for ministry in ministries:
            dossiers = get_first_n_dossiers_by_ministry(
                unique_foi_dossier_ids, 1, ministry
            )

            if i == 5:
                temp_dossiers = get_first_n_dossiers_by_ministry(
                    full_tuple, 5, ministry
                )
                for j in temp_dossiers:
                    if j not in dossiers:
                        dossiers.append(j)
                    if len(dossiers) == 5:
                        break

            nr_of_dossiers += len(dossiers)
            # Filter the DataFrame for the dossiers
            woo_data_filtered = filtered_df[filtered_df["foi_dossierId"].isin(dossiers)]
            woo_data_filtered_no_requests = woo_data_filtered[
                woo_data_filtered["documents_dc_type"] != "verzoek"
            ]

            result = pd.concat(
                [result, keep_relevant_columns(woo_data_filtered)], ignore_index=True
            )
            result_no_requests = pd.concat(
                [
                    result_no_requests,
                    keep_relevant_columns(woo_data_filtered_no_requests),
                ],
                ignore_index=True,
            )

        result_dir = f"{save_directory}/{content_folder_name}_{nr_of_dossiers}_dossiers"
        result_no_requests_dir = f"{save_directory}/{content_folder_name}_{nr_of_dossiers}_dossiers_no_requests"

        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(result_no_requests_dir, exist_ok=True)

        result.to_csv(
            f"{result_dir}/woo_merged.csv.gz", index=False, compression="gzip"
        )
        result_no_requests.to_csv(
            f"{result_no_requests_dir}/woo_merged.csv.gz",
            index=False,
            compression="gzip",
        )

    # Now make the complete subset with the relevant columns
    result = keep_relevant_columns(woo_data)
    result_no_requests = keep_relevant_columns(
        woo_data[woo_data["documents_dc_type"] != "verzoek"]
    )

    result_dir = f"{save_directory}/all_dossiers"
    result_no_requests_dir = f"{save_directory}/all_dossiers_no_requests"

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_no_requests_dir, exist_ok=True)

    result.to_csv(f"{result_dir}/woo_merged.csv.gz", index=False, compression="gzip")
    result.to_csv(
        f"{result_no_requests_dir}/woo_merged.csv.gz", index=False, compression="gzip"
    )
    print("[Info] ~ Subsets of ministries created.")


if __name__ == "__main__":
    main()
