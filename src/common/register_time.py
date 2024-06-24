import time
import os
from datetime import datetime


class Timer:
    def __init__(self, file_name, folder_name="time_logs"):
        self.file_path = os.path.join(folder_name, file_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        self.start_times = []
        self.elapsed_time = 0.0
        self.load_initial_time()

    def load_initial_time(self):
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
            print("[Info] ~ Creating time register file.", flush=True)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.start_times.append(current_time)
        self.start_time = time.time()

    def save_time(self):
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
