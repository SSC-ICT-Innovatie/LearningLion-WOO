import os
import pandas as pd

def save_docs(df:pd.DataFrame, filename:str) -> None:
    """
    Saves the given dataframe to the given filename. If the file already exists, it will not be overwritten.
    """
    if not os.path.exists(filename):
        df.to_csv(filename, compression='gzip')
    else:
        print(f"Skipping file as it already exists at location: {filename}")

# Selecting the paths and folder names
file = 'woo_merged.csv.gz'
folder = 'WoogleDumps_01-04-2024_12817_dossiers'
output_folder = 'WoogleDumps_01-04-2024_12817_dossiers_no_requests'
# path = f'/scratch/nju/docs'
path = f'../docs'   
bodytext_dataframe = pd.read_csv(f'{path}/{folder}/{file}', compression='gzip')

# Filtering the DataFrame
filtered_df = bodytext_dataframe[bodytext_dataframe['documents_dc_type'].str.lower() != 'verzoek']

if not os.path.exists(f'{path}/{output_folder}'):
    os.makedirs(f'{path}/{output_folder}')
save_docs(filtered_df, f'{path}/{output_folder}/{file}')
