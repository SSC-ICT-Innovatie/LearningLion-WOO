# This script will make a subset of the most recent woo-dossiers, of the 12 different ministries.
# It will take the 1, 5 and 20 most recent dossiers of each ministry that have a valid request file,
# resulting in a total of 12, 60 and 240 dossiers respectively. It will also only keep the necessary columns.

# Example with arguments:
# python preprocess_make_subset.py -c WoogleDumps_01-04-2024_12817_dossiers -d ./docs -s ./docs_ministries

import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser

def keep_relevant_columns(df):
    df['all_foi_bodyText'] = np.where(df['bodytext_foi_bodyTextOCR'].notnull() & df['bodytext_foi_bodyTextOCR'].str.strip().ne(''), df['bodytext_foi_bodyTextOCR'], df['bodytext_foi_bodyText'])

    # Only keep the necessary columns, and rename them
    rename_dict = {
        'id': 'page_id',
        'foi_documentId': 'document_id',
        'foi_dossierId': 'dossier_id',
        'all_foi_bodyText': 'bodyText',
        'documents_dc_type': 'type',
        'dossiers_dc_publisher_name': 'publisher',
        'documents_dc_source': 'source'
    }

    # New DataFrame with renamed and subset of columns
    return df.rename(columns=rename_dict)[['page_id', 'document_id', 'dossier_id', 'bodyText', 'type', 'publisher', 'source']]


def keep_relevant_columns(df):
    df['all_foi_bodyText'] = np.where(df['bodytext_foi_bodyTextOCR'].notnull() & df['bodytext_foi_bodyTextOCR'].str.strip().ne(''), df['bodytext_foi_bodyTextOCR'], df['bodytext_foi_bodyText'])

    # Only keep the necessary columns, and rename them
    rename_dict = {
        'id': 'page_id',
        'foi_documentId': 'document_id',
        'foi_dossierId': 'dossier_id',
        'all_foi_bodyText': 'bodyText',
        'documents_dc_type': 'type',
        'dossiers_dc_publisher_name': 'publisher',
        'documents_dc_source': 'source'
    }

    # New DataFrame with renamed and subset of columns
    return df.rename(columns=rename_dict)[['page_id', 'document_id', 'dossier_id', 'bodyText', 'type', 'publisher', 'source']]

def intersection_of_tuples(list1, list2):
    # Convert lists of tuples to sets
    set1 = set(list1)
    set2 = set(list2)

    # Find intersection
    intersection_set = set1 & set2

    # Convert the set back to a list of tuples
    return list(intersection_set)

def left_join_of_tuples(list1, list2):
    # Convert lists of tuples to sets
    set1 = set(list1)
    set2 = set(list2)

    # Find left join
    left_join_set = set1 - set2

    # Convert the set back to a list of tuples
    return list(left_join_set)

def make_unique_tuple(tuple_list):
    id_list = []
    unique_list = []
    for id, publisher  in tuple_list:
        if id not in id_list:
            id_list.append(id)
            unique_list.append((id, publisher))
    return unique_list

def get_first_n_dossiers_by_ministry(tuples, n, ministry):
    results = []
    for dossier_id, dossier_ministry in tuples:
        if dossier_ministry.lower() == ministry:
            results.append(dossier_id)
            if len(results) == n:
                break
    if len(results) < n:
        print(f"Warning: Only {len(results)} dossiers found for {ministry}, with n={n}.")
    return results

def main():
    parser = ArgumentParser(description="Document ingestion script using the Ingester class.")
    parser.add_argument('-c', '--content_folder_name', type=str)
    parser.add_argument('-d', '--documents_directory', type=str)
    parser.add_argument('-s', '--save_directory', type=str)
    args = parser.parse_args()

    if args.content_folder_name and args.documents_directory:
        content_folder_name = args.content_folder_name
        documents_directory = args.documents_directory
        save_directory = args.save_directory
    else:
        print("Not all arguments are provided")
        exit()
    print(f"Source folder of documents: {content_folder_name}")
    
    file_path = f'{documents_directory}/{content_folder_name}/woo_merged.csv.gz'
    woo_data = pd.read_csv(file_path, compression='gzip')

    filtered_df = woo_data[woo_data['dossiers_dc_publisher_name'].str.contains("Ministerie", case=False)]
    # Get all dossiers that are from ministries and have a request file
    filtered_df_request = woo_data[(woo_data['dossiers_dc_publisher_name'].str.contains("Ministerie", case=False)) & (woo_data['documents_dc_type'] == "verzoek")]
    dossier_ids_with_publishers_request = list(zip(filtered_df_request['foi_dossierId'], filtered_df_request['dossiers_dc_publisher_name']))
    dossier_tuple_request = make_unique_tuple(dossier_ids_with_publishers_request)

    # Get all dossiers that are from ministries and have an attachment file
    filtered_df_attachment = woo_data[(woo_data['dossiers_dc_publisher_name'].str.contains("Ministerie", case=False)) & (woo_data['documents_dc_type'] == "bijlage")]
    dossier_ids_with_publishers_attachment = list(zip(filtered_df_attachment['foi_dossierId'], filtered_df_attachment['dossiers_dc_publisher_name']))
    dossier_tuple_attachment = make_unique_tuple(dossier_ids_with_publishers_attachment)
    
    # Make a full tuple of all dossiers
    full_dossier_ids = list(zip(filtered_df_attachment['foi_dossierId'], filtered_df_attachment['dossiers_dc_publisher_name']))
    full_tuple = make_unique_tuple(full_dossier_ids)

    # Get the intersection of the two sets
    unique_foi_dossier_ids = intersection_of_tuples(dossier_tuple_request, dossier_tuple_attachment)
    unique_foi_dossier_ids
    
    full_tuple_left_join = left_join_of_tuples(full_tuple, unique_foi_dossier_ids)

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
        "ministerie van algemene zaken"
    ]

    number_of_dossiers = [1, 5]
    for i in number_of_dossiers:
        result = pd.DataFrame()
        result_no_requests = pd.DataFrame()
        nr_of_dossiers = 0
        for ministry in ministries:
            dossiers = get_first_n_dossiers_by_ministry(unique_foi_dossier_ids, 1, ministry)
            if i == 5:
                dossiers += get_first_n_dossiers_by_ministry(full_tuple_left_join, 4, ministry)
            nr_of_dossiers += len(dossiers)
            # Filter the DataFrame for the dossiers
            woo_data_filtered = filtered_df[filtered_df['foi_dossierId'].isin(dossiers)]
            woo_data_filtered_no_requests = woo_data_filtered[woo_data_filtered['documents_dc_type'] != "verzoek"]
            
            result = pd.concat([result, keep_relevant_columns(woo_data_filtered)], ignore_index=True)
            result_no_requests = pd.concat([result_no_requests, keep_relevant_columns(woo_data_filtered_no_requests)], ignore_index=True)
        
        result_dir = f"{save_directory}/{content_folder_name}_{nr_of_dossiers}_dossiers"
        result_no_requests_dir = f"{save_directory}/{content_folder_name}_{nr_of_dossiers}_dossiers_no_requests"
        
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(result_no_requests_dir, exist_ok=True)
        
        result.to_csv(f"{result_dir}/woo_merged.csv.gz", index=False, compression='gzip')
        result_no_requests.to_csv(f"{result_no_requests_dir}/woo_merged.csv.gz", index=False, compression='gzip')
        
    # Now make the complete subset with the relevant columns
    result = keep_relevant_columns(woo_data)
    result_no_requests = keep_relevant_columns(woo_data[woo_data['documents_dc_type'] != "verzoek"])
    
    result_dir = f"{save_directory}/all_dossiers"
    result_no_requests_dir = f"{save_directory}/all_dossiers_no_requests"
    
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(result_no_requests_dir, exist_ok=True)
    
    result.to_csv(f"{result_dir}/woo_merged.csv.gz", index=False, compression='gzip')
    result.to_csv(f"{result_no_requests_dir}/woo_merged.csv.gz", index=False, compression='gzip')
    print("Complete subset created.")

if __name__ == "__main__":
    main()