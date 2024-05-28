import settings
import utils
from argparse import ArgumentParser
from loguru import logger


class Ingester:
    def __init__(
        self,
        collection_name: str,
        content_folder: str,
        vectordb_folder: str,
        embeddings_provider=None,
        embeddings_model=None,
        text_splitter_method=None,
        vecdb_type=None,
        chunk_size=None,
        chunk_overlap=None,
        local_api_url=None,
        file_no=None,
        azureopenai_api_version=None,
        data_type=None,
    ):
        self.collection_name = collection_name
        self.content_folder = content_folder
        self.vectordb_folder = vectordb_folder
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        self.text_splitter_method = settings.TEXT_SPLITTER_METHOD if text_splitter_method is None else text_splitter_method
        self.vecdb_type = settings.VECDB_TYPE if vecdb_type is None else vecdb_type
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.local_api_url = settings.API_URL if local_api_url is None and settings.API_URL is not None else local_api_url
        self.file_no = file_no
        self.azureopenai_api_version = settings.AZUREOPENAI_API_VERSION if azureopenai_api_version is None and settings.AZUREOPENAI_API_VERSION is not None else azureopenai_api_version
        self.data_type = settings.DATA_TYPE if data_type is None else data_type


def main():
    """
    Creates an instance of Ingester class and ingests documents when necessary
    """
    parser = ArgumentParser(description="Document ingestion script using the Ingester class.")
    parser.add_argument("-p", "--embeddings_provider", type=str)
    parser.add_argument("-m", "--embeddings_model", type=str)
    parser.add_argument("-c", "--content_folder_name", type=str)

    args = parser.parse_args()
    # Get source folder with docs from user
    if args.content_folder_name:
        content_folder_name = args.content_folder_name
    else:
        content_folder_name = utils.get_content_folder_name(only_check_woo=settings.DATA_TYPE == "woo")
    logger.info(f"Source folder of documents: {content_folder_name}")

    # get associated source folder path and vectordb path
    content_folder_path, vectordb_folder_path = utils.create_vectordb_name(content_folder_name)
    ingester = Ingester(
        collection_name=content_folder_name,
        content_folder=content_folder_path,
        vectordb_folder=vectordb_folder_path,
        embeddings_provider=args.embeddings_provider,
        embeddings_model=args.embeddings_model,
    )
    ingester.ingest()
    logger.info(f"finished ingesting documents for folder {content_folder_name}")


if __name__ == "__main__":
    main()
