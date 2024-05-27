import heapq
import json
import nltk
import numpy as np
import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi, BM25L, BM25Plus

# If necessary, download the NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text: str, index: int=0, print_progress:bool=True, print_freq:int=100) -> list[str]:
    if print_progress and index and index % print_freq == 0:
        print(f"Processing document {index}")
    
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
    
def check_relevance(ground_truth, retrieved) -> int:
    # Check if the retrieved documents are relevant
    return len(retrieved.intersection(ground_truth))

# Selecting the paths
folder = 'WoogleDumps_01-04-2024_10_dossiers_no_requests'
# input_path = f'/scratch/nju/docs/{folder}/woo_merged.csv.gz'
input_path = f'../docs/{folder}/woo_merged.csv.gz'
evaluation_json = 'evaluation_request_WoogleDumps_01-04-2024_10_dossiers_no_requests.json'

# Only set the necessary data type
dtypes={"bodytext_foi_bodyText": str, "bodytext_foi_bodyTextOCR": str}
woo_data = pd.read_csv(input_path, dtype=dtypes, compression='gzip')

# Merge all the different pages of a document into one row, and remove double whitespace
woo_data['all_foi_bodyText'] = np.where(
    woo_data['bodytext_foi_bodyText'].notnull() & woo_data['bodytext_foi_bodyText'].str.strip().ne(''), 
    woo_data['bodytext_foi_bodyText'], 
    woo_data['bodytext_foi_bodyTextOCR']
)

# Group by foi_documentId and aggregate both all_foi_bodyText and dossiers_dc_title
grouped_bodyText = woo_data.groupby('foi_documentId').agg({
    'all_foi_bodyText': lambda x: ' '.join(x.astype(str)),
    'dossiers_dc_title': 'first',
    'id': 'first',
    'foi_dossierId': 'first'
}).reset_index()

# Generate corpus, which is a list of all the words per document
corpus = grouped_bodyText['all_foi_bodyText'].tolist()

print(f"Number of documents in corpus: {len(corpus)}")

# Do preprocessing for echt document
tokenized_corpus = [preprocess_text(doc, i) for i, doc in enumerate(corpus)]

print(tokenized_corpus[:1], flush=True)

# Set the BM25 algorithms
bm25okapi = BM25Okapi(tokenized_corpus)
bm25l = BM25L(tokenized_corpus)
bm25plus = BM25Plus(tokenized_corpus)
all_bm25 = [bm25okapi, bm25l, bm25plus]

with open(f"../evaluation/{evaluation_json}", 'r') as file:
    json_data = json.load(file)
for bm25 in all_bm25:
    # Execute it twice to get the top n_pages, n_documents, and n_dossiers and 1
    for i in range(2):
        results_raw = {}
        results = pd.DataFrame(columns=['#Relevant Pages', '#Relevant Pages Retrieved', '#Relevant Documents', '#Relevant Documents Retrieved', '#Relevant Dossiers', '#Relevant Dossiers Retrieved'])
        for key, value in json_data.items():
            if not value.get('pages'):
                print("No pages found in the JSON file")
                continue
            if not value.get('documents'):
                print("No documents found in the JSON file")
                continue
            if not value.get('dossier'):
                print("No dossiers found in the JSON file")
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
            
            pages_result = [grouped_bodyText['id'][i] for i in n_pages_result]
            documents_result = [grouped_bodyText['foi_documentId'][i] for i in n_documents_result]
            dossiers_result = [grouped_bodyText['foi_dossierId'][i] for i in n_dossiers_result]
            
            print(pages_result)
            exit()
            
            results_raw[key] = {
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
            results.loc[len(results.index)] = new_row

        # Saving results DataFrame to a CSV file
        base_directory = '../evaluation/results'
        csv_file_name = f'{evaluation_json.split(".")[0]}_{"1" if i == 1 else "n"}_{bm25.__class__.__name__}_request.csv'
        csv_file_path = os.path.join(base_directory, csv_file_name)

        # Check if the /results directory exists, if not, create it
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)

        results.to_csv(csv_file_path, index=False)

        # Saving results_raw dictionary to a JSON file
        json_file_name = f'{evaluation_json.split(".")[0]}_{"1" if i == 1 else "n"}_{bm25.__class__.__name__}_request_raw.json'
        json_file_path = os.path.join(base_directory, json_file_name)

        with open(json_file_path, 'w') as file:
            json.dump(results_raw, file)