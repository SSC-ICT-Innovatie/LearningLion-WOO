# This script will generate an ground truth evaluation file for the requests in the woo-dossiers.
# The script will generate a JSON file with the following structure:
# { "bodytext": {"pages": [page1, page2, ...], "documents": [document1, document2, ...], "dossier": [dossierId] }

# Examples with arguments:
# python create_evaluation_file.py -c WoogleDumps_01-04-2024_12817_dossiers_12_dossiers -d ./docs -e ./evaluation
# python create_evaluation_file.py -c 12_dossiers -d ./docs -e ./evaluation

import json
import os
import pandas as pd
from argparse import ArgumentParser


def main():
    # Parse all the arguments and read the settings
    parser = ArgumentParser()
    parser.add_argument("-c", "--content_folder_name", type=str)
    parser.add_argument("-d", "--documents_directory", type=str)
    parser.add_argument("-e", "--evaluation_directory", type=str)
    args = parser.parse_args()

    if (
        args.content_folder_name
        and args.documents_directory
        and args.evaluation_directory
    ):
        content_folder_name = args.content_folder_name
        documents_directory = args.documents_directory
        evaluation_directory = args.evaluation_directory
    else:
        raise ValueError("Please provide all arguments.")
    print(f"[Info] ~ Source folder of documents: {content_folder_name}")

    file_path = f"{documents_directory}/{content_folder_name}/woo_merged.csv.gz"
    woo_data = pd.read_csv(file_path, compression="gzip")

    # Find all requests
    requests = woo_data[woo_data["type"].str.lower() == "verzoek"]
    print("[Info] ~ Length requests: ", len(requests))

    # Get dataframe without the requests
    no_requests_dataframe = woo_data[woo_data["type"].str.lower() != "verzoek"]
    print("[Info] ~ Length no requests df: ", len(no_requests_dataframe))

    # Get the aggregated text for each dossier
    # Structure: { foi_dossierId: bodytext_foi_bodyTextOCR }
    aggregated_requests = (
        requests.groupby("dossier_id")["bodyText"]
        .apply(lambda texts: " ".join(map(str, texts)))
        .to_dict()
    )

    # Create ground truth
    # Structure: { foi_dossierId: { pages: [page1, page2, ...], documents: [document1, document2, ...] } }
    aggregated_dict = (
        no_requests_dataframe.groupby("dossier_id")
        .apply(
            lambda x: {
                "pages": list(x["page_id"].unique()),
                "documents": list(x["document_id"].unique()),
            }
        )
        .to_dict()
    )

    # Merge bodytext and ground truth
    # Structure: { bodytext: { pages: [page1, page2, ...], documents: [document1, document2, ...], dossier: [dossierId] } }
    merged_structure = {}
    for dossier_id, body_text in aggregated_requests.items():
        normalized_body_text = " ".join(body_text.split())
        if dossier_id in aggregated_dict:
            merged_structure[normalized_body_text] = {
                "pages": aggregated_dict[dossier_id]["pages"],
                "documents": aggregated_dict[dossier_id]["documents"],
                "dossier": [
                    dossier_id
                ],  # Encapsulating dossier_id in a list as per your requirement
            }

    json_file_path = os.path.join(
        evaluation_directory,
        f"evaluation_request_{content_folder_name}_no_requests.json",
    )

    # Check if the directory exists, if not, create it
    if not os.path.exists(evaluation_directory):
        os.makedirs(evaluation_directory)

    # Write the aggregated JSON to a file
    with open(json_file_path, "w") as file:
        json.dump(merged_structure, file)


if __name__ == "__main__":
    main()
