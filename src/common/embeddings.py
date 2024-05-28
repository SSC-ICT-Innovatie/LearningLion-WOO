from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings


def getEmbeddings(embeddings_provider: str, embeddings_model_name: str) -> Embeddings:
    """
    Retrieves embeddings based on the specified provider and model name.

    Args:
        embeddings_provider (str): The provider of the embeddings. Must be one of ["local_embeddings"].
        embeddings_model_name (str): The name of the embeddings model.

    Returns:
        Embeddings: The retrieved embeddings.
    """
    if embeddings_provider == "local_embeddings":
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        if embeddings_model_name == "meta-llama/Meta-Llama-3-8B-Instruct" or embeddings_model_name == "meta-llama/Meta-Llama-3-8B":
            # Llama3 needs a padding token for the tokenizer.
            # More info: https://www.reddit.com/r/LangChain/comments/16m1nee/comment/kxw4bfb/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
            tokenizer = embeddings.client.tokenizer
            tokenizer.pad_token = tokenizer.eos_token

        print(f"[Info] ~ Loaded local embeddings: {embeddings_model_name}", flush=True)
    else:
        raise ValueError(f"Unknown embeddings provider: {embeddings_provider}")
    return embeddings
