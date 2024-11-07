# LearningLion-WOO

The project is a study on the use of generative AI to improve the services of SSC-ICT by supporting employees and optimizing internal processes. Originally, the focus is on generative large language models (LLM), in the form of Retrieval Augmented Generation (RAG), because they can have the most significant impact on the daily work of SSC-ICT employees. This version dipes deeper into the Retrieval part in RAG. The original version can be found [here](https://github.com/SSC-ICT-Innovatie/LearningLion).


This version serves as part of the Master Thesis of [Nicky Ju](https://github.com/JuNicky).

The paper corresponding to this repository can be found in the TU Delft Repository.


## Flow Chart
![Flow Chart](https://github.com/SSC-ICT-Innovatie/LearningLion-WOO/tree/main/!%20project_docs/images/LearningLion-woo-workflow.jpg)


## Files
Filenames starting with 
- create --> create evaluation files with specific preprocessing
- evaluate --> running queries on vector database/corpus
- ingest --> creating vector database/corpus
- preprocess --> preprocess the data in different ways before creating the database
- relevance --> (re-)evaluating the results


## Complete Example Pipeline
This guide assumes that you are familiar with the basics of Python (such as setting up environment, and installing packages).

1. First steps
	- Have your data dump downloaded from Woogle: [Dump of 19/04/2024](https://ssh.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-zau-e3rk) or [Daily updated dump (password protected)](http://surfdrive.surf.nl/files/index.php/s/NEpv6uiFwvigxqx/authenticate).
	- Merge all the data using `merge_woo.ipynb`.
    - Create evaluation files `create_evaluation_file.py` or with `create_evaluation_file_keywords_paraphrase.ipynb`.
2. Preprocess Data
	- Run preprocess `preprocess_real_words.py` or `preprocess_stem_stopwords.py` to preprocess the data in different ways.
3. Database creation
    - Create Vector Store with `ingest_embeddings.py`.
    - Create BM25 Corpus with `ingest_bm25.py`.
4. Evaluation
	- Run the evaluation files with the vector store/bm25 corpus `evaluate_bm25.py` or `evaluate_embeddings.py`.
5. Evaluation metrics
    - `relevance_evaluation.ipynb` to calculate basic metrics like precision and recall.
	- `relevance_dossier_average.ipynb` for frequency based, `relevance_dossier_MAP.ipynb` for weighted frequency based.
	