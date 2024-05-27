# Examples with arguments:
# python evaluate_bm25_document_similarity.py --content_folder_name 12_dossiers_no_requests --documents_directory ./docs --dataset_folder_name 12_dossiers_no_requests --results_folder ./evaluation/results
# python evaluate_bm25_document_similarity.py --content_folder_name 12_dossiers_no_requests --documents_directory ./docs --dataset_folder_name 60_dossiers_no_requests --results_folder ./evaluation/results

import heapq
import pandas as pd
from argparse import ArgumentParser
from rank_bm25 import BM25Okapi
from common import evaluate_helpers


def main():
    parser = ArgumentParser()
    parser.add_argument("--content_folder_name", type=str)
    parser.add_argument("--documents_directory", type=str)
    parser.add_argument("--dataset_folder_name", type=str)
    parser.add_argument("--results_folder", type=str)
    args = parser.parse_args()

    if (
        args.content_folder_name
        and args.documents_directory
        and args.dataset_folder_name
        and args.results_folder
    ):
        content_folder_name = args.content_folder_name
        documents_directory = args.documents_directory
        dataset_folder_name = args.dataset_folder_name
        results_folder = args.results_folder
    else:
        raise ValueError("Please provide all arguments.")

    woo_data = pd.read_csv(
        f"./{documents_directory}/{content_folder_name}/woo_merged.csv.gz"
    )

    # Generate corpus, which is a list of all the words per document
    corpus = woo_data["bodyText"].tolist()

    print(f"Number of documents in corpus: {len(corpus)}", flush=True)

    # Do preprocessing for echt document
    tokenized_corpus = [evaluate_helpers.preprocess_text(doc) for doc in corpus]
    bm25okapi = BM25Okapi(tokenized_corpus)

    result = pd.DataFrame(
        columns=[
            "page_id",
            "dossier_id",
            "retrieved_page_ids",
            "retrieved_dossier_ids",
            "scores",
            "number_of_correct_dossiers",
            "dossier#1",
            "dossier#2",
            "dossier#3",
            "dossier#4",
            "dossier#5",
            "dossier#6",
            "dossier#7",
            "dossier#8",
            "dossier#9",
            "dossier#10",
            "dossier#11",
            "dossier#12",
            "dossier#13",
            "dossier#14",
            "dossier#15",
            "dossier#16",
            "dossier#17",
            "dossier#18",
            "dossier#19",
            "dossier#20",
        ]
    )

    woo_data_subset = pd.read_csv(
        f"./{documents_directory}/{dataset_folder_name}/woo_merged.csv.gz"
    )
    for _, row in woo_data_subset.iterrows():
        if pd.isna(row["bodyText"]):
            continue
        tokenized_query = evaluate_helpers.tokenize(row["bodyText"])
        doc_scores = bm25okapi.get_scores(tokenized_query)

        n_pages_result = heapq.nlargest(
            21, range(len(doc_scores)), key=doc_scores.__getitem__
        )
        retrieved_page_ids = []
        retrieved_dossier_ids = []
        # scores = []
        for i in n_pages_result:
            if woo_data["page_id"][i] == row["page_id"]:
                print("[Info] ~ Same document retrieved", flush=True)
                continue
            if len(retrieved_page_ids) == 20:
                print("[Info] ~ 20 documents retrieved", flush=True)
                break
            retrieved_page_ids.append(woo_data["page_id"][i])
            retrieved_dossier_ids.append(woo_data["dossier_id"][i])
        new_row = {
            "page_id": row["page_id"],
            "dossier_id": row["dossier_id"],
            "retrieved_page_ids": ", ".join(retrieved_page_ids),
            "retrieved_dossier_ids": ", ".join(retrieved_dossier_ids),
            "scores": "",
            "number_of_correct_dossiers": retrieved_dossier_ids.count(
                row["dossier_id"]
            ),
            "dossier#1": retrieved_dossier_ids[0] == row["dossier_id"],
            "dossier#2": retrieved_dossier_ids[1] == row["dossier_id"],
            "dossier#3": retrieved_dossier_ids[2] == row["dossier_id"],
            "dossier#4": retrieved_dossier_ids[3] == row["dossier_id"],
            "dossier#5": retrieved_dossier_ids[4] == row["dossier_id"],
            "dossier#6": retrieved_dossier_ids[5] == row["dossier_id"],
            "dossier#7": retrieved_dossier_ids[6] == row["dossier_id"],
            "dossier#8": retrieved_dossier_ids[7] == row["dossier_id"],
            "dossier#9": retrieved_dossier_ids[8] == row["dossier_id"],
            "dossier#10": retrieved_dossier_ids[9] == row["dossier_id"],
            "dossier#11": retrieved_dossier_ids[10] == row["dossier_id"],
            "dossier#12": retrieved_dossier_ids[11] == row["dossier_id"],
            "dossier#13": retrieved_dossier_ids[12] == row["dossier_id"],
            "dossier#14": retrieved_dossier_ids[13] == row["dossier_id"],
            "dossier#15": retrieved_dossier_ids[14] == row["dossier_id"],
            "dossier#16": retrieved_dossier_ids[15] == row["dossier_id"],
            "dossier#17": retrieved_dossier_ids[16] == row["dossier_id"],
            "dossier#18": retrieved_dossier_ids[17] == row["dossier_id"],
            "dossier#19": retrieved_dossier_ids[18] == row["dossier_id"],
            "dossier#20": retrieved_dossier_ids[19] == row["dossier_id"],
        }

        # Append the new row to the DataFrame
        result.loc[len(result)] = new_row

    result.to_csv(
        f"{results_folder}/document_similarity_{content_folder_name}_{dataset_folder_name}_bm25.csv"
    )
    print(
        f"[Info] ~ Result BM25 document similarity for {content_folder_name} saved.",
        flush=True,
    )


if __name__ == "__main__":
    main()
