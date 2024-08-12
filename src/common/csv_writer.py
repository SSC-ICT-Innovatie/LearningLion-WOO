import csv
import os


class CSVWriter:
    def __init__(self, content_folder_name, algorithm, evaluation_file="", document_similarity=False, folder_name="./evaluation/results"):
        self.file_name = self._generate_file_name(content_folder_name, algorithm, evaluation_file, document_similarity)
        self.file_path = os.path.join(folder_name, self.file_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        self.last_index = self._determine_last_index()
        self.csv_file = None
        self.csv_writer = None
        self._open()

    def _generate_file_name(self, content_folder_name, algorithm, evaluation_file, document_similarity):
        if "/" in algorithm:
            algorithm = algorithm.split("/")[-1]
        if len(evaluation_file) != 0:
            evaluation_file = evaluation_file.split(".")[0]
            return f"evaluation_{content_folder_name}_{evaluation_file}_{algorithm}.csv"
        if document_similarity:
            return f"document_similarity_{content_folder_name}_{algorithm}.csv"
        raise ValueError("Document similarity or evaluation not specified.")

    def _determine_last_index(self):
        """Determine the last index processed in the CSV file if it exists."""
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                reader = csv.reader(file)
                return sum(1 for row in reader)
        return -1

    def _open(self):
        """Open the CSV file for appending or writing, setting up the writer."""
        self.csv_file = open(self.file_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        if self.last_index == -1:
            self.write_header()

    def write_header(self):
        """Write the CSV header."""
        self.csv_writer.writerow(
            [
                "page_id",
                "dossier_id",
                "retrieved_page_ids",
                "retrieved_dossier_ids",
                "scores",
                "precision",
                "recall",
                "map",
                "number_of_correct_dossiers",
                *(f"dossier#{i+1}" for i in range(50)),
            ]
        )

    def write_row(self, row_data):
        """Write a row to the CSV file."""
        self.csv_writer.writerow(row_data)

    def close(self):
        """Close the CSV file."""
        if self.csv_file:
            self.csv_file.close()

    def get_last_index(self):
        """Get the last processed index."""
        return self.last_index
