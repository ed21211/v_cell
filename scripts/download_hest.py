import os
import pandas as pd
from huggingface_hub import snapshot_download

REPO_ID = "MahmoodLab/hest"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(SCRIPT_DIR, "..", "datasets", "hest_ccrcc")
SAMPLE_CSV = os.path.join(SCRIPT_DIR, "..", "datasets", "hest_metadata", "ccrcc_samples.csv")

TEST_SINGLE_SAMPLE = True  # set False once you've confirmed the pattern matches correctly


def build_allow_patterns(sample_ids):
    patterns = []
    for sid in sample_ids:
        patterns.append(f"*{sid}*")   # broad match; refine once you see actual folder layout
    return patterns


def main():
    df = pd.read_csv(SAMPLE_CSV)
    sample_ids = df["id"].astype(str).tolist()

    if TEST_SINGLE_SAMPLE:
        sample_ids = sample_ids[:1]
        print(f"TEST MODE: downloading single sample only: {sample_ids}")
    else:
        print(f"Downloading {len(sample_ids)} ccRCC samples: {sample_ids}")

    patterns = build_allow_patterns(sample_ids)
    print(f"Allow patterns: {patterns}")

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