import nltk
import os
import pandas as pd
import re
import settings
from argparse import ArgumentParser
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
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")
        return ""

    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words and stem
    processed_words = [stemmer.stem(word) for word in tokens if word not in stop_words]

    # Join the words back into a single string with spaces
    return " ".join(processed_words)

def main():
    parser = ArgumentParser(description="Document ingestion script using the Ingester class.")
    parser.add_argument('-c', '--content_folder_name', type=str)
    parser.add_argument('-d', '--documents_directory', type=str)
    
    args = parser.parse_args()
    # Get source folder with docs from user
    if args.content_folder_name:
        content_folder_name = args.content_folder_name
    else:
        print("Please provide the source folder of documents.")
        exit()
    print(f"Source folder of documents: {content_folder_name}")
    documents_directory = args.documents_directory if args.documents_directory else settings.DOC_DIR
    
    # Selecting the paths
    input_path = f'{documents_directory}/{content_folder_name}/woo_merged.csv.gz'

    # Only set the necessary data type
    woo_data = pd.read_csv(input_path, compression='gzip')
    print(f"Number of documents in corpus: {len(woo_data)}")

    # Do preprocessing for every document
    woo_data['bodyText'] = woo_data.apply(lambda row: preprocess_text(row['bodyText'], row.name, True, 100), axis=1)

    # Save results to CSV file
    # Create the directory if it does not exist
    output_dir = os.path.join(documents_directory, f'{content_folder_name}_stopwords')
    os.makedirs(output_dir, exist_ok=True)

    # Save the processed data back to a compressed CSV file
    woo_data.to_csv(f'{output_dir}/woo_merged.csv.gz', compression='gzip', index=False)

if __name__ == "__main__":
    main()