# import os
# import zipfile

# from huggingface_hub import snapshot_download
# from tqdm import tqdm
# import pandas as pd


# def download_hest(patterns, local_dir):
#     repo_id = 'MahmoodLab/hest'
#     snapshot_download(repo_id=repo_id, allow_patterns=patterns, repo_type="dataset", local_dir=local_dir)

#     seg_dir = os.path.join(local_dir, 'cellvit_seg')
#     if os.path.exists(seg_dir):
#         print('Unzipping cell vit segmentation...')
#         for filename in tqdm([s for s in os.listdir(seg_dir) if s.endswith('.zip')]):
#             path_zip = os.path.join(seg_dir, filename)
                        
#             with zipfile.ZipFile(path_zip, 'r') as zip_ref:
#                 zip_ref.extractall(seg_dir)


# # # Note that the full dataset is around 1TB of data
# # download_hest('*', local_dir)


# # To download a subset

# local_dir='datasets/hest_data' # hest will be dowloaded to this folder

# ids_to_query = ['TENX96', 'TENX99'] # list of ids to query

# list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
# download_hest(list_patterns, local_dir) # see method definition above


# # Query HEST by organ, techonology, oncotree code...

# meta_df = pd.read_csv("hf://datasets/MahmoodLab/hest/HEST_v1_3_0.csv")


# print(meta_df['organ'].unique())
# print(meta_df['oncotree_code'].unique())

# # # Filter the dataframe by organ, oncotree code...
# # meta_df = meta_df[meta_df['oncotree_code'] == 'IDC']
# # meta_df = meta_df[meta_df['organ'] == 'Breast']

# # ids_to_query = meta_df['id'].values

# # list_patterns = [f"*{id}[_.]**" for id in ids_to_query]
# # download_hest(list_patterns, local_dir) # see method definition above


"""
Step 2: Download only the ccRCC samples from HEST-1k, using the IDs
found in ccrcc_samples.csv (produced by inspect_hest_metadata.py).

Run this on a machine with HF access.
"""
import os
import pandas as pd
from huggingface_hub import snapshot_download

REPO_ID = "MahmoodLab/hest"
LOCAL_DIR = "../datasets/hest_metadata"
SAMPLE_CSV = "../datasets/hest_metadata/ccrcc_samples.csv"


def build_allow_patterns(sample_ids):
    """HEST-1k stores per-sample files (WSI, expression, metadata) typically
    keyed by sample id across several subfolders (e.g. wsis/, st/, metadata/).
    We build patterns that match each sample id in any subfolder/extension."""
    patterns = []
    for sid in sample_ids:
        patterns.append(f"*{sid}*")   # broad match; refine once you see actual folder layout
    return patterns


def main():
    df = pd.read_csv(SAMPLE_CSV)
    sample_ids = df["id"].astype(str).tolist()
    print(f"Downloading {len(sample_ids)} ccRCC samples: {sample_ids}")

    patterns = build_allow_patterns(sample_ids)

    os.makedirs(LOCAL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=LOCAL_DIR,
    )
    print(f"\nDone. Data downloaded to: {LOCAL_DIR}")


if __name__ == "__main__":
    main()