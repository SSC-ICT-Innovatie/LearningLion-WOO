from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from typing import List, Tuple, Dict, Any


def get_chroma_vector_store(collection_name: str, embeddings: Embeddings, vectordb_folder: str) -> Chroma:
    """
    Creates and returns a Chroma vector store.

    Args:
        collection_name (str): The name of the collection.
        embeddings (Embeddings): The embedding function.
        vectordb_folder (str): The folder to persist the vector store.

    Returns:
        Chroma: The Chroma vector store.
    """
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=vectordb_folder,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"[Info] ~ Loaded Chroma vector store for {collection_name}", flush=True)
    return vector_store


def get_documents_with_scores(vector_store: VectorStore, question: str, search_kwargs: Dict[str, Any] = {"k": 100}) -> List[Tuple[Document, float]]:
    """
    Performs a similarity search in the vector store for documents related to the query, returning the top results.

    Parameters:
        vector_store (VectorStore): The vector store containing document vectors.
        question (str): The query string for the search.
        search_kwargs (Dict[str, Any]): Optional parameters for the search (default is {"k": 100}).

    Returns:
        List[Tuple[Document, float]]: List of document-score tuples for the top relevant documents.
    """
    return vector_store.similarity_search_with_relevance_scores(question, **search_kwargs)
