# Example with arguments:
# python evaluate_bm25.py -a BM25Okapi -c WoogleDumps_01-04-2024_10_dossiers_no_requests_fake_stopwords -d ../docs -e evaluation_request_WoogleDumps_01-04-2024_10_dossiers_no_requests.json
# python evaluate_bm25.py -c 12_dossiers_no_requests -d ../docs_ministries -e evaluation_request_Ministries_12_dossiers_no_requests.json

import heapq
import json
import nltk
import os
import pandas as pd
import re
from argparse import ArgumentParser
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi, BM25L, BM25Plus

def preprocess_text(text: str, index: int=0, print_progress: bool=True, print_freq:int=100) -> list[str]:
    if print_progress and index and index % print_freq == 0:
        print(f"Processing document {index}", flush=True)
    
    # Initialize stop words and stemmer
    stop_words = set(stopwords.words('dutch'))
    stemmer = PorterStemmer()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove unnecessary whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words and stem
    return [stemmer.stem(word) for word in tokens if word not in stop_words]

def tokenize(text) -> list[str]:
    # Check if text is of type string
    if not isinstance(text, str):
        return []
    # Tokenize the text
    return word_tokenize(text)
    
def check_relevance(ground_truth, retrieved) -> int:
    # Check if the retrieved documents are relevant
    return len(retrieved.intersection(ground_truth))


def calculate_precision_recall(retrieved, relevant):
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    true_positives = len(retrieved_set.intersection(relevant_set))
    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(relevant_set) if relevant_set else 0
    return precision, recall

def unique_ordered_list(original_list: list, take_head:int=-1) -> list:
    """
    This function removes duplicates from a list while preserving the original order.

    Parameters:
        original_list (list): The list from which duplicates need to be removed.

    Returns:
        list: A list containing only unique elements, maintaining the order of first appearance.
    """
    unique_list = []
    seen = set()

    for element in original_list:
        if element not in seen:
            unique_list.append(element)
            seen.add(element)

    if take_head > 0:
        return unique_list[:take_head]
    return unique_list
    

def run_bm25(woo_data, bm25, evaluation, evaluation_file, content_folder_name, corpus_length):
    document_identifiers = []
    dossier_identifiers = []

    # Loop through each key in the JSON data to extract documents
    for key in evaluation:
        document_identifiers.extend(evaluation[key].get("documents", []))
        dossier_identifiers.extend(evaluation[key].get("dossier", []))
        
    print(f"Number of documents in evaluation: {len(document_identifiers)}", flush=True)
    print(f"Number of dossiers in evaluation: {len(dossier_identifiers)}", flush=True)
    
    # Determine file paths
    csv_file_path = f'../evaluation/results/{evaluation_file.split(".")[0]}_{content_folder_name}_{bm25.__class__.__name__}_request.xlsx'
    last_index = -1
    
    # Check if csv file exists
    # csv_file_exists = os.path.exists(csv_file_path)
    # csv_file = open(csv_file_path, 'w')
    # csv_writer = None
    
    columns = ['#Retrieved', '#Relevant Documents', '#Relevant Documents Retrieved', '#Relevant Dossiers', '#Relevant Dossiers Retrieved', 'Precision Document', 'Recall Document', 'Precision Dossier', 'Recall Dossier']
    
    # TODO, read the csv file if it already exists
    # Initialize an empty list to store the DataFrames
    dfs = []

    # Loop to create n DataFrames
    for _ in range(len(dossier_identifiers)):
        df = pd.DataFrame(0,  index=range(len(document_identifiers)), columns=columns)
        dfs.append(df)
    
    for index, (key, value) in enumerate(evaluation.items()):
        if index <= last_index:
            print(f"Skipping index {index}", flush=True)
            continue
        if not value.get('pages'):
            print("No pages found in the JSON file", flush=True)
            continue
        if not value.get('documents'):
            print("No documents found in the JSON file", flush=True)
            continue
        if not value.get('dossier'):
            print("No dossiers found in the JSON file", flush=True)
            continue


        tokenized_query = preprocess_text(key)
        doc_scores = bm25.get_scores(tokenized_query)
        
        print(doc_scores)
        
        # Assuming n_pages >= n_documents >= n_dossiers
        n_pages_result = heapq.nlargest(corpus_length, range(len(doc_scores)), key=doc_scores.__getitem__)
        
        # for i, col in enumerate(df.columns[1:]):  # Skip the first column ('id')
        for i in range(len(document_identifiers)):
            # results = n_pages_result[:i+1]
            documents_result = unique_ordered_list([woo_data['document_id'][j] for j in n_pages_result], i + 1)
            dossiers_result = unique_ordered_list([woo_data['dossier_id'][j] for j in n_pages_result], i + 1)
            values_to_add = [
                check_relevance(set(value['documents']), set(documents_result)),
                check_relevance(set(value['dossier']), set(dossiers_result))
            ]
            for col in [2, 4]:
                dfs[index].loc[i, columns[col]] = values_to_add[int(col / 2) - 1]

        
        # Set the columns that are already known
        dfs[index].loc[:, columns[0]] = range(1, len(document_identifiers)+1)
        dfs[index].loc[:, columns[1]] = len(value.get('documents'))
        dfs[index].loc[:, columns[3]] = len(value.get('dossier'))
        
        # Set precision and recall
        TP_document = dfs[index].loc[:, columns[2]].to_numpy().flatten()
        FP_document = dfs[index].loc[:, columns[0]].to_numpy().flatten() - TP_document
        FN_document = dfs[index].loc[:, columns[1]].to_numpy().flatten() - TP_document
        TP_dossier = dfs[index].loc[:, columns[4]].to_numpy().flatten()
        FP_dossier = dfs[index].loc[:, columns[0]].to_numpy().flatten() - TP_dossier
        FN_dossier = dfs[index].loc[:, columns[3]].to_numpy().flatten() - TP_dossier

        dfs[index].loc[:, columns[5]] = TP_document / (TP_document + FP_document)
        dfs[index].loc[:, columns[6]] = TP_document / (TP_document + FN_document)
        dfs[index].loc[:, columns[7]] = TP_dossier / (TP_dossier + FP_dossier)
        dfs[index].loc[:, columns[8]] = TP_dossier / (TP_dossier + FN_dossier)
    
        # Using ExcelWriter to write multiple sheets
    with pd.ExcelWriter(csv_file_path, engine='openpyxl') as writer:
        for index, df in enumerate(dfs):
            df.to_excel(writer, sheet_name=f'Sheet{index+1}', index=False)

        # if csv_writer is None:
        #     csv_writer = pd.DataFrame(columns=columns)
        # if not csv_file_exists:
        #     csv_writer.to_csv(csv_file, index=False, lineterminator='\n')
        #     csv_file_exists = True  # Prevent header repetition
        # csv_writer.loc[len(csv_writer.index)] = new_row
        # csv_writer.loc[len(csv_writer.index)-1:len(csv_writer.index)-1].to_csv(csv_file, header=False, index=False, lineterminator='\n')
        # csv_file.flush() # Ensure that the data is written to the file in the DHPC environment
        # print(f"Index {index} in csv file written.", flush=True)
            
    # csv_file.close()


def main():
    # If necessary, download the NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    print("Successfully downloaded the NLTK resources.", flush=True)
    
    parser = ArgumentParser()
    parser.add_argument('-a', '--algorithm', type=str)
    parser.add_argument('-c', '--content_folder_name', type=str)
    parser.add_argument('-d', '--documents_directory', type=str)
    parser.add_argument('-e', '--evaluation_file', type=str)
    parser.add_argument('-r', '--retrieve_whole_document', type=bool, default=False)
    
    args = parser.parse_args()
    if args.content_folder_name and args.documents_directory and args.evaluation_file:
        content_folder_name = args.content_folder_name
        documents_directory = args.documents_directory
        evaluation_file = args.evaluation_file
        retrieve_whole_document = args.retrieve_whole_document
        if args.algorithm in ["BM25Okapi", "BM25L", "BM25Plus"]:
            algorithm = args.algorithm
        else:
            algorithm = "all"
    else:
        print("Please provide the source folder of documents, the output folder name, and the database directory.", flush=True)
        exit()
    print(f"Source folder of documents: {content_folder_name}", flush=True)

    
    # Selecting the paths
    file_name = "woo_merged.csv.gz"
    input_path = f'{documents_directory}/{content_folder_name}/{file_name}'
    evaluation_path = f'../evaluation/{evaluation_file}'

    woo_data = pd.read_csv(input_path, compression='gzip')
    
    # Preprocess woo data, merge all the documents into one entry
    if retrieve_whole_document:
        woo_data = woo_data.groupby('document_id').agg({
            'bodyText': lambda x: ' '.join(x.astype(str)),
            'page_id': 'first',
            'dossier_id': 'first',
        }).reset_index()

    # Generate corpus, which is a list of all the words per document
    corpus = woo_data['bodyText'].tolist()

    print(f"Number of documents in corpus: {len(corpus)}", flush=True)

    # Do preprocessing for echt document
    tokenized_corpus = [tokenize(doc) for doc in corpus]
    
    with open(evaluation_path, 'r') as file:
        evaluation = json.load(file)
        
    print(f"Number of documents in evaluation: {len(evaluation)}", flush=True)
    
    if algorithm == "BM25Okapi" or algorithm == "all":
        print("Starting BM25Okapi", flush=True)
        bm25okapi = BM25Okapi(tokenized_corpus)
        run_bm25(woo_data, bm25okapi, evaluation, evaluation_file, content_folder_name, len(corpus))
        print("BM25Okapi done", flush=True)
    if algorithm == "BM25L" or algorithm == "all":
        print("Starting BM25L", flush=True)
        bm25l = BM25L(tokenized_corpus)
        run_bm25(woo_data, bm25l, evaluation, evaluation_file, content_folder_name, len(corpus))
        print("BM25L done", flush=True)
    if algorithm == "BM25Plus" or algorithm == "all":
        print("Starting BM25Plus", flush=True)
        bm25plus = BM25Plus(tokenized_corpus)
        run_bm25(woo_data, bm25plus, evaluation, evaluation_file, content_folder_name, len(corpus))
        print("BM25Plus done", flush=True)

if __name__ == "__main__":
    print("Starting the program...", flush=True)
    main()