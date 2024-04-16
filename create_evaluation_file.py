import json
import os
import pandas as pd

# Selecting the paths and folder names
folder = 'WoogleDumps_01-04-2024_12817_dossiers'
# input_path = f'/scratch/nju/docs/{folder}/woo_merged.csv.gz'
input_path = f'../docs/{folder}/woo_merged.csv.gz'
bodytext_dataframe = pd.read_csv(input_path, compression='gzip')

# Find all requests
requests = bodytext_dataframe[bodytext_dataframe['documents_dc_type'].str.lower() == 'verzoek']
print("Length requests: ", len(requests))

# Get dataframe without the requests
no_requests_dataframe = bodytext_dataframe[~bodytext_dataframe['documents_dc_type'].str.lower().isin(['verzoek', 'verzoeken'])]
print("Length no requests df: ", len(no_requests_dataframe))

# Get the aggregated text for each dossier
# Structure: { foi_dossierId: bodytext_foi_bodyTextOCR }
aggregated_requests = (
    requests.groupby('foi_dossierId')['bodytext_foi_bodyTextOCR']
    .apply(lambda texts: ' '.join(map(str, texts)))
    .to_dict()
)

# Create ground truth
# Structure: { foi_dossierId: { pages: [page1, page2, ...], documents: [document1, document2, ...] } }
aggregated_dict = (
    no_requests_dataframe.groupby('foi_dossierId')
    .apply(lambda x: {
        "pages": list(x['id'].unique()),
        "documents": list(x['foi_documentId'].unique()),
    })
    .to_dict()
)

# Merge bodytext and ground truth
# Structure: { bodytext: { pages: [page1, page2, ...], documents: [document1, document2, ...], dossier: [dossierId] } }
merged_structure = {}
for dossier_id, body_text in aggregated_requests.items():
    if dossier_id in aggregated_dict:
        merged_structure[body_text] = {
            "pages": aggregated_dict[dossier_id]["pages"],
            "documents": aggregated_dict[dossier_id]["documents"],
            "dossier": [dossier_id]  # Encapsulating dossier_id in a list as per your requirement
        }

directory = '../evaluation'
json_file_path = os.path.join(directory, f'evaluation_request_{folder}_no_requests.json')

# Check if the directory exists, if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)

# Write the aggregated JSON to a file
with open(json_file_path, 'w') as file:
    json.dump(merged_structure, file)
    