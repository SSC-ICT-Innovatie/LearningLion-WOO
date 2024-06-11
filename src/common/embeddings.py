from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings


def getEmbeddings(embedding_model_name: str) -> Embeddings:
    """
    Retrieves embeddings based on the specified provider and model name.

    Args:
        embedding_model_name (str): The name of the embeddings model.

    Returns:
        Embeddings: The retrieved embeddings.
    """
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    if embedding_model_name in ["meta-llama/Meta-Llama-3-8B-Instruct", "meta-llama/Meta-Llama-3-8B"]:
        # Llama3 needs a padding token for the tokenizer.
        # More info: https://www.reddit.com/r/LangChain/comments/16m1nee/comment/kxw4bfb/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
        tokenizer = embeddings.client.tokenizer
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[Info] ~ Loaded local embeddings: {embedding_model_name}", flush=True)
    return embeddings
