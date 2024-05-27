import settings
import utils
from argparse import ArgumentParser
from ingest.ingester import Ingester
from loguru import logger



def main():
    '''
        Creates an instance of Ingester class and ingests documents when necessary
    '''
    parser = ArgumentParser(description="Document ingestion script using the Ingester class.")
    parser.add_argument('-p', '--embeddings_provider', type=str)
    parser.add_argument('-m', '--embeddings_model', type=str)
    parser.add_argument('-c', '--content_folder_name', type=str)
    
    args = parser.parse_args()
    # Get source folder with docs from user
    if args.content_folder_name:
        content_folder_name = args.content_folder_name
    else:
        content_folder_name = utils.get_content_folder_name(only_check_woo=settings.DATA_TYPE == "woo")
    logger.info(f"Source folder of documents: {content_folder_name}")
    
    # get associated source folder path and vectordb path
    content_folder_path, vectordb_folder_path = utils.create_vectordb_name(content_folder_name)
    ingester = Ingester(collection_name=content_folder_name, content_folder=content_folder_path, vectordb_folder=vectordb_folder_path, embeddings_provider=args.embeddings_provider, embeddings_model=args.embeddings_model)
    ingester.ingest()
    logger.info(f"finished ingesting documents for folder {content_folder_name}")
     
if __name__ == "__main__":
    main()
