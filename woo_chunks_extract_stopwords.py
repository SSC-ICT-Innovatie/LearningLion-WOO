import nltk
import numpy as np
import os
import pandas as pd
import re
import settings
from argparse import ArgumentParser
from ingest.ingest_utils import IngestUtils
from ingest.woo_parser import WooParser
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# If necessary, download the NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text: str, index: int=0, print_progress:bool=True, print_freq:int=100) -> list[str]:
    if print_progress and index and index % print_freq == 0:
        print(f"Processing document {index}")
    
    # Initialize stop words and stemmer
    stop_words = set(stopwords.words('dutch'))
    stemmer = PorterStemmer()
    
    try:
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remove unnecessary whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
    except:
        return []

    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words and stem
    return [stemmer.stem(word) for word in tokens if word not in stop_words]

def main():
    parser = ArgumentParser(description="Document ingestion script using the Ingester class.")
    parser.add_argument('-c', '--content_folder_name', type=str)
    
    args = parser.parse_args()
    # Get source folder with docs from user
    if args.content_folder_name:
        content_folder_name = args.content_folder_name
    else:
        print("Please provide the source folder of documents.")
        exit()
    print(f"Source folder of documents: {content_folder_name}")
    
    # Selecting the paths
    # folder = 'WoogleDumps_01-04-2024_12817_dossiers_no_requests'
    input_path = f'/scratch/nju/docs/{content_folder_name}/woo_merged.csv.gz'
    # input_path = f'../docs/{folder}/woo_merged.csv.gz'
    # evaluation_json = 'evaluation_request_WoogleDumps_01-04-2024_12817_dossiers_no_requests.json'

    # Only set the necessary data type
    # dtypes={"bodytext_foi_bodyTextOCR": str}
    woo_data = pd.read_csv(input_path, compression='gzip')

    if 'bodytext_foi_bodyText' in woo_data.columns:
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
    else:
        corpus = woo_data['foi_bodyTextOCR'].tolist()
        
    print(f"Number of documents in corpus: {len(corpus)}")

    # Do preprocessing for every document
    tokenized_corpus = [preprocess_text(doc, i) for i, doc in enumerate(corpus)]
    
    # Saving processed data
    with open(f'{content_folder_name}_processed_corpus.txt', 'w') as file:
        for document in tokenized_corpus:
            file.write(' '.join(document) + '\n')
            
if __name__ == "__main__":
    main()