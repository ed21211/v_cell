import argparse
import os

import h5py
import anndata as ad
import tifffile


def decode_barcode_value(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def main():
    parser = argparse.ArgumentParser(description="Check a HEST sample structure and H&E-ST alignment.")
    parser.add_argument("--sample-id", type=str, required=True, help="Example: INT1, TENX105, etc.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Example: ../datasets/hest_ccrcc or ../datasets/hest_skin_cancer",
    )
    args = parser.parse_args()

    sample_id = args.sample_id
    data_dir = args.data_dir

    paths = {
        "wsi": f"{data_dir}/wsis/{sample_id}.tif",
        "st": f"{data_dir}/st/{sample_id}.h5ad",
        "patches": f"{data_dir}/patches/{sample_id}.h5",
        "metadata": f"{data_dir}/metadata/{sample_id}.json",
    }

    print(f"Checking sample: {sample_id}")
    print(f"Data dir: {data_dir}\n")

    # 1. Check files exist
    missing = []
    for name, path in paths.items():
        print(f"{name}: {path}")
        exists = os.path.exists(path)
        print("  exists:", exists)

        if exists:
            print("  size MB:", round(os.path.getsize(path) / 1e6, 2))
        else:
            missing.append(name)

        print()

    if missing:
        print(f"WARNING: missing files: {missing}")
        print("Stopping early because core files are missing.")
        return

    # 2. Check WSI opens
    print("Checking WSI...")
    with tifffile.TiffFile(paths["wsi"]) as tif:
        print("  number of pages:", len(tif.pages))
        print("  first page shape:", tif.pages[0].shape)
        print("  dtype:", tif.pages[0].dtype)

    # 3. Check h5ad opens
    print("\nChecking ST h5ad...")
    adata = ad.read_h5ad(paths["st"])
    print("  adata shape:", adata.shape)
    print("  obs columns:", list(adata.obs.columns)[:20])
    print("  var columns:", list(adata.var.columns)[:20])
    print("  obsm keys:", list(adata.obsm.keys()))
    print("  uns keys:", list(adata.uns.keys()))

    if "spatial" in adata.obsm:
        print("  spatial coordinates shape:", adata.obsm["spatial"].shape)
        print("  first 5 spatial coords:")
        print(adata.obsm["spatial"][:5])
    else:
        print("  WARNING: no adata.obsm['spatial'] found")

    # 4. Check patches h5
    print("\nChecking patches h5...")
    with h5py.File(paths["patches"], "r") as f:
        print("  keys:", list(f.keys()))

        def show_h5(name, obj):
            if hasattr(obj, "shape"):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")

        f.visititems(show_h5)

    # 5. Basic alignment sanity check
    print("\nBasic alignment check...")
    print("  number of ST spots/cells:", adata.n_obs)

    with h5py.File(paths["patches"], "r") as f:
        for key in f.keys():
            obj = f[key]
            if hasattr(obj, "shape"):
                print(f"  patches file dataset '{key}' shape:", obj.shape)

    # 6. Matching fields
    print("\nChecking possible patch/ST matching fields...")

    with h5py.File(paths["patches"], "r") as f:
        keys = list(f.keys())

        for possible_key in ["barcode", "barcodes", "coords", "coordinates", "spatial"]:
            if possible_key in keys:
                data = f[possible_key][()]
                print(f"  Found {possible_key}: shape={data.shape}")

                if len(data) == adata.n_obs:
                    print(f"  {possible_key} count matches adata.n_obs")

    print("  adata obs_names example:")
    print(adata.obs_names[:5].tolist())

    # 7. Barcode overlap
    print("\nChecking barcode overlap...")

    with h5py.File(paths["patches"], "r") as f:
        if "barcode" not in f.keys():
            print("  WARNING: no 'barcode' dataset in patches file")
            return

        patch_barcodes_raw = f["barcode"][:]

    patch_barcodes = []
    for b in patch_barcodes_raw:
        # Handles shape (N, 1) or flat shape (N,)
        if hasattr(b, "__len__") and not isinstance(b, bytes):
            value = b[0]
        else:
            value = b

        patch_barcodes.append(decode_barcode_value(value))

    adata_barcodes = set(adata.obs_names.astype(str))
    overlap = [b for b in patch_barcodes if b in adata_barcodes]

    print("  number of patch barcodes:", len(patch_barcodes))
    print("  number of adata obs names:", len(adata_barcodes))
    print("  overlap:", len(overlap))
    print("  first 10 patch barcodes:", patch_barcodes[:10])
    print("  first 10 overlapping barcodes:", overlap[:10])

    if len(overlap) == 0:
        print("  WARNING: no barcode overlap found")
    elif len(overlap) < len(patch_barcodes):
        print("  NOTE: some patches do not overlap with ST obs_names")
    else:
        print("  Barcode overlap looks good.")

    print("\nDone.")


if __name__ == "__main__":
    main()