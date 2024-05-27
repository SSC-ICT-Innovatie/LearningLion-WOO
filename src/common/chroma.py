from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings


def get_chroma_vector_store(
    collection_name: str, embeddings: Embeddings, vectordb_folder: str
) -> Chroma:
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
