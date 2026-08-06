import os
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "MahmoodLab/hest"
METADATA_FILE = "HEST_v1_3_0.csv"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "datasets", "hest_metadata")


def load_metadata(filename):
    path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename)
    df = pd.read_csv(path)
    print(f"\nLoaded metadata: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", list(df.columns))
    return df


def summarize_organs(df):
    for col in ["organ", "tissue", "oncotree_code", "disease_state", "st_technology", "dataset_title"]:
        if col in df.columns:
            print(f"\n--- value_counts for '{col}' ---")
            print(df[col].value_counts(dropna=False).head(50))


def summarize_subset(df, name):
    print(f"\n=== {name} ===")
    print(f"Number of samples: {len(df)}")

    for col in ["organ", "tissue", "oncotree_code", "disease_state", "st_technology", "patient", "species", "dataset_title"]:
        if col in df.columns:
            print(f"\n--- {name}: value_counts for '{col}' ---")
            print(df[col].value_counts(dropna=False).head(50))

    preview_cols = [
        "id",
        "organ",
        "tissue",
        "oncotree_code",
        "disease_state",
        "st_technology",
        "patient",
        "species",
        "dataset_title",
    ]
    preview_cols = [c for c in preview_cols if c in df.columns]

    print(f"\n--- {name}: sample preview ---")
    print(df[preview_cols].to_string(index=False))


def save_subset(df, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df)} samples to {path}")


def filter_by_organ(df, organ_name):
    mask = df["organ"].astype(str).str.lower().eq(organ_name.lower())
    filtered = df[mask].copy()
    print(f"\nFound {len(filtered)} rows where organ == {organ_name}")
    return filtered


def filter_by_disease_state(df, states):
    states_lower = [s.lower() for s in states]
    mask = df["disease_state"].astype(str).str.lower().isin(states_lower)
    filtered = df[mask].copy()
    print(f"\nFound {len(filtered)} rows matching disease_state in {states}")
    return filtered


def filter_by_oncotree(df, codes):
    """Filter rows by exact oncotree_code match."""
    codes_lower = [c.lower() for c in codes]
    mask = df["oncotree_code"].astype(str).str.lower().isin(codes_lower)
    filtered = df[mask].copy()
    print(f"\nFound {len(filtered)} rows matching oncotree_code in {codes}")
    return filtered


def filter_by_st_technology(df, technology):
    mask = df["st_technology"].astype(str).str.lower().eq(technology.lower())
    filtered = df[mask].copy()
    print(f"\nFound {len(filtered)} rows where st_technology == {technology}")
    return filtered


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_metadata(METADATA_FILE)
    summarize_organs(df)

    print("\n=== ALL SKIN SAMPLES ===")
    skin_all_df = filter_by_organ(df, "Skin")
    summarize_subset(skin_all_df, "all_skin")
    save_subset(skin_all_df, "skin_all_samples.csv")

    print("\n=== SKIN CANCER BY DISEASE STATE ===")
    cancer_states = ["Cancer", "Tumor", "Cancer/Genetically modified"]
    skin_cancer_df = filter_by_disease_state(skin_all_df, cancer_states)
    summarize_subset(skin_cancer_df, "skin_cancer_by_disease_state")
    save_subset(skin_cancer_df, "skin_cancer_samples.csv")

    print("\n=== SKIN / MELANOMA RELATED BY ONCOTREE ===")
    skin_oncotree_codes = ["SKCM", "MEL", "CSCC"]
    skin_melanoma_df = filter_by_oncotree(df, skin_oncotree_codes)
    summarize_subset(skin_melanoma_df, "skin_melanoma_related")
    save_subset(skin_melanoma_df, "skin_melanoma_related_samples.csv")

    print("\n=== SKIN CANCER VISIUM ONLY ===")
    skin_cancer_visium_df = filter_by_st_technology(skin_melanoma_df, "Visium")
    summarize_subset(skin_cancer_visium_df, "skin_cancer_visium")
    save_subset(skin_cancer_visium_df, "skin_cancer_visium_samples.csv")

    print("\nSUMMARY")
    print(f"All skin samples:               {len(skin_all_df)}")
    print(f"Skin cancer by disease_state:   {len(skin_cancer_df)}")
    print(f"Skin/melanoma by OncoTree:      {len(skin_melanoma_df)}")
    print(f"Skin cancer Visium only:        {len(skin_cancer_visium_df)}")