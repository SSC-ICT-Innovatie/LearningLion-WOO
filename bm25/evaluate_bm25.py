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

def preprocess_text(text: str, index: int=0, print_progress:bool=True, print_freq:int=100) -> list[str]:
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

def tokenize(text: str) -> list[str]:
    # Tokenize the text
    return word_tokenize(text)
    
def check_relevance(ground_truth, retrieved) -> int:
    # Check if the retrieved documents are relevant
    return len(retrieved.intersection(ground_truth))

def run_bm25(woo_data, bm25, evaluation, evaluation_file, content_folder_name):
    # Check if chunks are present in the data
    evaluate_chunks = 'chunks_id' in woo_data.columns
    # Execute it twice to get the top n_pages, n_documents, and n_dossiers and 1 respectively
    for i in range(2):
        print(f"Running algorithm: {bm25.__class__.__name__}, with i = {'1' if i == 1 else 'n'}", flush=True)
        
        # Determine file paths
        csv_file_path = f'../evaluation/results/{evaluation_file.split(".")[0]}_{"1" if i == 1 else "n"}_{content_folder_name}_{bm25.__class__.__name__}_request.csv'
        json_file_path = f'../evaluation/results/{evaluation_file.split(".")[0]}_{"1" if i == 1 else "n"}_{content_folder_name}_{bm25.__class__.__name__}_request_raw.json'
        last_index = -1
        
        # Check if csv file exists
        csv_file_exists = os.path.exists(csv_file_path)
        csv_file = open(csv_file_path, 'a')
        csv_writer = None
        
        # Check if JSON file exists
        json_file_exists = os.path.exists(json_file_path)
        if not json_file_exists:
            json_file = open(json_file_path, 'w')
            json_file.write('[]')
        else:
            with open(json_file_path) as f:
                json_file = json.load(f)
                last_index_string = list(json_file[-1].keys())[0]
                # Find a key to attempt to convert to integer (assuming there is at least one key)
                if last_index_string:
                    # Convert the last key to integer (will raise ValueError if not possible)
                    last_index = int(list(last_index_string)[-1])
                    print(f"Starting with index: {last_index}", flush=True)
                else:
                    print("Error: The last dictionary is empty, no keys to convert.", flush=True)
                    exit()

        for index, (key, value) in enumerate(evaluation.items()):
            if index <= last_index:
                print(f"Skipping index {index}", flush=True)
                continue
            results_raw = {}
            if not value.get('pages'):
                print("No pages found in the JSON file", flush=True)
                continue
            if not value.get('documents'):
                print("No documents found in the JSON file", flush=True)
                continue
            if not value.get('dossier'):
                print("No dossiers found in the JSON file", flush=True)
                continue

            # Assuming n == len(value)
            if i == 0:
                n_pages = len(value['pages'])
                n_documents = len(value['documents'])
                n_dossiers = len(value['dossier'])
            else:
                n_pages = 1
                n_documents = 1
                n_dossiers = 1

            tokenized_query = preprocess_text(key)
            doc_scores = bm25.get_scores(tokenized_query)
            
            print(doc_scores)
            print(doc_scores.__getitem__)
            
            # Assuming n_pages >= n_documents >= n_dossiers
            n_pages_result = heapq.nlargest(n_pages, range(len(doc_scores)), key=doc_scores.__getitem__)
            n_documents_result = n_pages_result[:n_documents]
            n_dossiers_result = n_pages_result[:n_dossiers]        
            print(n_pages_result)
            
            chunks_result = [woo_data['chunks_id'][i] for i in n_pages_result] if evaluate_chunks else None
            pages_result = [woo_data['page_id'][i] for i in n_pages_result]
            documents_result = [woo_data['document_id'][i] for i in n_documents_result]
            dossiers_result = [woo_data['dossier_id'][i] for i in n_dossiers_result]
            
            print(pages_result)
            exit()

            results_raw[index] = {
                'bodyText': key,
                'chunks': chunks_result,
                'pages': pages_result,
                'documents': documents_result,
                'dossier': dossiers_result
            }
            
            # Collect top documents and their scores for the current BM25 algorithm
            new_row = [
                len(value['pages']), #Relevant Pages
                check_relevance(set(value['pages']), set(pages_result)), #Relevant Pages Retrieved
                len(value['documents']), #Relevant Documents
                check_relevance(set(value['documents']), set(documents_result)), #Relevant Documents Retrieved
                len(value['dossier']), #Relevant Dossiers
                check_relevance(set(value['dossier']), set(dossiers_result)), #Relevant Dossiers Retrieved
            ]

            if csv_writer is None:
                fieldnames = ['#Relevant Pages', '#Relevant Pages Retrieved', '#Relevant Documents', '#Relevant Documents Retrieved', '#Relevant Dossiers', '#Relevant Dossiers Retrieved']
                csv_writer = pd.DataFrame(columns=fieldnames)
            if not csv_file_exists:
                csv_writer.to_csv(csv_file, index=False, lineterminator='\n')
                csv_file_exists = True  # Prevent header repetition
            csv_writer.loc[len(csv_writer.index)] = new_row
            csv_writer.loc[len(csv_writer.index)-1:len(csv_writer.index)-1].to_csv(csv_file, header=False, index=False, lineterminator='\n')
            csv_file.flush() # Ensure that the data is written to the file in the DHPC environment
            print(f"Index {index} in csv file written.", flush=True)
                
            # Append or create JSON file
            with open(json_file_path) as json_file:
                all_results_raw = json.load(json_file)
                all_results_raw.append(results_raw)
            with open(json_file_path, 'w') as json_file:
                json.dump(all_results_raw, json_file)
            print(f"Index {index} in json file written.", flush=True)
                
        csv_file.close()

        # # Saving results DataFrame to a CSV file
        # base_directory = './evaluation/results'
        # csv_file_name = f'{evaluation_file.split(".")[0]}_{"1" if i == 1 else "n"}_{content_folder_name}_{bm25.__class__.__name__}_request.csv'
        # csv_file_path = os.path.join(base_directory, csv_file_name)

        # # Check if the /results directory exists, if not, create it
        # if not os.path.exists(base_directory):
        #     os.makedirs(base_directory)

        # results.to_csv(csv_file_path, index=False)

        # # Saving results_raw dictionary to a JSON file
        # json_file_name = f'{evaluation_file.split(".")[0]}_{"1" if i == 1 else "n"}_{content_folder_name}_{bm25.__class__.__name__}_request_raw.json'
        # json_file_path = os.path.join(base_directory, json_file_name)

        # with open(json_file_path, 'w') as file:
        #     json.dump(results_raw, file)


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
    
    args = parser.parse_args()
    if args.content_folder_name and args.documents_directory and args.evaluation_file:
        content_folder_name = args.content_folder_name
        documents_directory = args.documents_directory
        evaluation_file = args.evaluation_file
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
    print(tokenized_corpus[:1], flush=True)
    
    with open(evaluation_path, 'r') as file:
        evaluation = json.load(file)
        
    print(f"Number of documents in evaluation: {len(evaluation)}", flush=True)
    
    if algorithm == "BM25Okapi" or algorithm == "all":
        print("Starting BM25Okapi", flush=True)
        bm25okapi = BM25Okapi(tokenized_corpus)
        run_bm25(woo_data, bm25okapi, evaluation, evaluation_file, content_folder_name)
        print("BM25Okapi done", flush=True)
    if algorithm == "BM25L" or algorithm == "all":
        print("Starting BM25L", flush=True)
        bm25l = BM25L(tokenized_corpus)
        run_bm25(woo_data, bm25l, evaluation, evaluation_file, content_folder_name)
        print("BM25L done", flush=True)
    if algorithm == "BM25Plus" or algorithm == "all":
        print("Starting BM25Plus", flush=True)
        bm25plus = BM25Plus(tokenized_corpus)
        run_bm25(woo_data, bm25plus, evaluation, evaluation_file, content_folder_name)
        print("BM25Plus done", flush=True)

if __name__ == "__main__":
    print("Starting the program...", flush=True)
    main()