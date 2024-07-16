"""
Time tracker class
First line contains the start times of all sessions.
Second line contains the estimated total time taken in seconds (float).
"""

import time
import os
from datetime import datetime


class Timer:
    def __init__(self, content_folder_name, algorithm, evaluation_file="", document_similarity=False, ingest=False, folder_name="./evaluation/results"):
        self.file_name = self._generate_file_name(content_folder_name, algorithm, evaluation_file, document_similarity, ingest)
        self.file_path = os.path.join(folder_name, self.file_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        self.start_times = []
        self.elapsed_time = 0.0
        self._load_initial_time()

    def _load_initial_time(self):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.start_time = time.time()
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                lines = file.readlines()
                if len(lines) == 2:
                    try:
                        self.start_times = lines[0].strip().split(", ")
                        self.elapsed_time = float(lines[1].strip())
                    except ValueError:
                        self.start_times = []
                        self.elapsed_time = 0.0
                        print("[Warning] ~ Time register file content is not valid. Starting fresh.", flush=True)
        else:
            print(f"[Info] ~ Creating time register file at location: {self.file_path}.", flush=True)
            start_times_str = ", ".join(self.start_times)
            with open(self.file_path, "w") as file:
                file.write(f"{start_times_str}\n{0}")
        self.start_times.append(current_time)
        print(f"[Info] ~ Starting time register with time: {current_time}", flush=True)

    def _generate_file_name(self, content_folder_name, algorithm, evaluation_file, document_similarity, ingest):
        if "/" in algorithm:
            algorithm = algorithm.split("/")[-1]
        if len(evaluation_file) != 0:
            evaluation_file = evaluation_file.split(".")[0]
            return f"evaluation_{content_folder_name}_{evaluation_file}_{algorithm}_time.txt"
        if document_similarity:
            return f"document_similarity_{content_folder_name}_{algorithm}_time.txt"
        if ingest:
            return f"ingest_{content_folder_name}_{algorithm}_time.txt"
        raise ValueError("Document similarity, evaluation or ingest not specified.")

    def update_time(self):
        if self.start_time is not None:
            elapsed_since_start = time.time() - self.start_time
            temporary_elapsed_time = self.elapsed_time + elapsed_since_start
        else:
            temporary_elapsed_time = self.elapsed_time

        if self.start_times:
            start_times_str = ", ".join(self.start_times)
            with open(self.file_path, "w") as file:
                file.write(f"{start_times_str}\n{temporary_elapsed_time}")
        else:
            raise ValueError("No start times to save.")
        
    def get_current_duration(self):
        if self.start_time is not None:
            elapsed_since_start = time.time() - self.start_time
            return self.elapsed_time + elapsed_since_start
        else:
            return self.elapsed_time
