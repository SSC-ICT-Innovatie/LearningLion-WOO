def generate_name(algorithm: str, content_folder_name: str, whole_document: bool=False, real_words: bool=False) -> str:
    name = f"{content_folder_name}_{algorithm}"
    if whole_document:
        name += "_whole_document"
    if real_words:
        name += "_real_words"
    return name