import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def preprocess_text(
    text: str, index: int = 0, print_progress: bool = False, print_freq: int = 100
) -> list[str]:
    """
    Preprocesses the input text by removing punctuation, unnecessary spaces, stop words,
    and applying stemming. Optionally, it can print progress for document processing.

    Parameters:
    text (str): The text to preprocess.
    index (int, optional): The index of the current document, used for progress tracking. Default is 0.
    print_progress (bool, optional): If set to True, prints the progress of text processing. Default is True.
    print_freq (int, optional): Frequency of progress messages, in terms of number of documents. Default is 100.

    Returns:
    list[str]: A list of processed tokens from the input text.
    """
    if type(text) != str:
        print("[Warning] ~ text is not of type str", flush=True)
        return []
    if print_progress and index and index % print_freq == 0:
        print(f"[Info] ~ Processing document {index}", flush=True)

    # Initialize stop words and stemmer
    stop_words = set(stopwords.words("dutch"))
    stemmer = PorterStemmer()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Remove unnecessary whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words and stem
    return [stemmer.stem(word) for word in tokens if word not in stop_words]


def tokenize(text: str) -> list[str]:
    """
    Tokenizes the input text into words using NLTK's word_tokenize.

    Parameters:
    text (str): The text to tokenize.

    Returns:
    list[str]: A list of tokens derived from the input text.
    """
    # Check if text is of type string
    if not isinstance(text, str):
        return []
    # Tokenize the text
    return word_tokenize(text)


def check_relevance(ground_truth: set, retrieved: set) -> int:
    """
    Calculates the number of relevant items in the retrieved set.

    Parameters:
    ground_truth (set): The set of ground truth items.
    retrieved (set): The set of retrieved items.

    Returns:
    int: The number of relevant items in the retrieved set.
    """
    return len(retrieved.intersection(ground_truth))
