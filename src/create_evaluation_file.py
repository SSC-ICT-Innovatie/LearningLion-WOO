"""
This script will generate an ground truth evaluation file for the requests in the woo-dossiers.
It will generate a JSON file with the following structure:
{ "bodytext": {"pages": [page1, page2, ...], "documents": [document1, document2, ...], "dossier": [dossierId] }

Examples with arguments:
python create_evaluation_file.py --content_folder_name minbzk --documents_directory ./final_docs_minbzk --evaluation_directory ./final_evaluation_minbzk
python create_evaluation_file.py --content_folder_name minbzk --documents_directory ./final_docs_minbzk --evaluation_directory ./final_evaluation_minbzk --real_words
"""

import json
import os
import pandas as pd
from argparse import ArgumentParser


def filter_body_text(text, words_set):
    # Split text into words and filter them
    filtered_words = [word for word in text.split() if word in words_set]
    # Join the filtered words back into a string
    return ' '.join(filtered_words)

def main():
    # Parse all the arguments and read the settings
    parser = ArgumentParser()
    parser.add_argument("--content_folder_name", type=str, required=True)
    parser.add_argument("--documents_directory", type=str, required=True)
    parser.add_argument("--evaluation_directory", type=str, required=True)
    parser.add_argument("--real_words", action="store_true")
    args = parser.parse_args()

    content_folder_name = args.content_folder_name
    documents_directory = args.documents_directory
    evaluation_directory = args.evaluation_directory
    print(f"[Info] ~ Source folder of documents: {content_folder_name}")

    file_path = f"{documents_directory}/{content_folder_name}/woo_merged.csv.gz"
    woo_data = pd.read_csv(file_path, compression="gzip")

    # Check if each dossier_id group has at least one 'verzoek' or 'besluit'
    has_verzoek = woo_data[woo_data["type"].str.lower() == "verzoek"].groupby("dossier_id").size() > 0
    has_besluit = woo_data[woo_data["type"].str.lower() == "besluit"].groupby("dossier_id").size() > 0

    # Check if each dossier_id group has at least one 'bijlage'
    has_bijlage = woo_data[woo_data["type"].str.lower() == "bijlage"].groupby("dossier_id").size() > 0

    # Combine the two masks to find dossier_ids that meet both conditions
    valid_dossier_ids_verzoek = has_verzoek[has_verzoek].index.intersection(has_bijlage[has_bijlage].index)
    valid_dossier_ids_besluit = has_besluit[has_besluit].index.intersection(has_bijlage[has_bijlage].index)

    valid_dossier_ids = list(valid_dossier_ids_verzoek.union(valid_dossier_ids_besluit))

    # Find all requests
    filtered_data = woo_data[woo_data["dossier_id"].isin(valid_dossier_ids)]
    print("[Info] ~ Length filtered df: ", len(filtered_data))

    # Get dataframe without the requests
    no_requests_filtered_dataframe = filtered_data[(filtered_data["type"].str.lower() != "verzoek") & (filtered_data["type"].str.lower() != "besluit")]
    print("[Info] ~ Length no requests filtered df: ", len(no_requests_filtered_dataframe))

    # Get the aggregated text for each dossier
    # Structure: { foi_dossierId: bodytext_foi_bodyTextOCR }
    aggregated_requests = filtered_data.groupby("dossier_id")["bodyText"].apply(lambda texts: " ".join(map(str, texts))).to_dict()

    # Create ground truth
    # Structure: { foi_dossierId: { pages: [page1, page2, ...], documents: [document1, document2, ...] } }
    aggregated_dict = (
        no_requests_filtered_dataframe.groupby("dossier_id")
        .apply(
            lambda x: {
                "pages": list(x["page_id"].unique()),
                "documents": list(x["document_id"].unique()),
            }
        )
        .to_dict()
    )

    if args.real_words:
        words_set = set()
        with open("./common/stopwords/wordlist-ascii.txt", 'r') as file:
            for line in file:
                words_set.add(line.strip())

    # Merge bodytext and ground truth
    # Structure: { bodytext: { pages: [page1, page2, ...], documents: [document1, document2, ...], dossier: [dossierId] } }
    merged_structure = {}
    for dossier_id, body_text in aggregated_requests.items():
        normalized_body_text = " ".join(body_text.split())
        if args.real_words:
            normalized_body_text = filter_body_text(str(normalized_body_text), words_set)
        if dossier_id in aggregated_dict:
            merged_structure[normalized_body_text] = {
                "pages": aggregated_dict[dossier_id]["pages"],
                "documents": aggregated_dict[dossier_id]["documents"],
                "dossier": [dossier_id],
            }

    # Counting all pages in the merged_structure
    total_pages_count = sum(len(details["pages"]) for details in merged_structure.values())

    print(f"[Info] ~ Data for {content_folder_name}:")
    print(f"[Info] ~ Length of the data: {len(merged_structure)}")
    print(f"[Info] ~ Length of #Pages: {total_pages_count}")

    json_file_path = os.path.join(
        evaluation_directory,
        f"evaluation_request_{content_folder_name}.json",
    )

    # Check if the directory exists, if not, create it
    if not os.path.exists(evaluation_directory):
        os.makedirs(evaluation_directory)

    # Write the aggregated JSON to a file
    with open(json_file_path, "w") as file:
        json.dump(merged_structure, file)


if __name__ == "__main__":
    main()
