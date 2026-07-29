from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
PRED_CSV = PROJECT_DIR / "outputs" / "hest_factor_baseline" / "ridge_baseline_predictions.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "hest_factor_baseline" / "maps"

FACTOR_NAMES = [
    "t_cell_infiltration",
    "cytotoxicity",
    "tgfb_caf",
    "proliferation",
]


def plot_sample_factor(df_sample, sample_id, factor, out_dir):
    true_col = f"true_{factor}"
    pred_col = f"pred_{factor}"

    if true_col not in df_sample.columns or pred_col not in df_sample.columns:
        print(f"Missing columns for {factor} in {sample_id}")
        return

    # shared color range for fair visual comparison
    vmin = min(df_sample[true_col].min(), df_sample[pred_col].min())
    vmax = max(df_sample[true_col].max(), df_sample[pred_col].max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc1 = axes[0].scatter(
        df_sample["spatial_x"],
        df_sample["spatial_y"],
        c=df_sample[true_col],
        s=8,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    axes[0].set_title(f"{sample_id} - TRUE {factor}")
    axes[0].set_xlabel("spatial_x")
    axes[0].set_ylabel("spatial_y")
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal")

    sc2 = axes[1].scatter(
        df_sample["spatial_x"],
        df_sample["spatial_y"],
        c=df_sample[pred_col],
        s=8,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
    )
    axes[1].set_title(f"{sample_id} - PRED {factor}")
    axes[1].set_xlabel("spatial_x")
    axes[1].set_ylabel("spatial_y")
    axes[1].invert_yaxis()
    axes[1].set_aspect("equal")

    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label(factor)

    fig.suptitle(f"{sample_id}: true vs predicted {factor}", fontsize=14)
    fig.tight_layout()

    out_path = out_dir / f"{sample_id}_{factor}_true_vs_pred.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", help="train / val / test / all")
    parser.add_argument("--sample-id", type=str, default=None, help="Optional single sample, e.g. INT22")
    args = parser.parse_args()

    print(f"Loading predictions from: {PRED_CSV}")
    df = pd.read_csv(PRED_CSV)

    print("Columns:", list(df.columns))
    print("Shape:", df.shape)

    if args.split != "all":
        df = df[df["split"] == args.split].copy()

    if args.sample_id is not None:
        df = df[df["sample_id"] == args.sample_id].copy()

    if df.empty:
        raise ValueError("No rows left after filtering. Check split/sample-id.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sample_ids = sorted(df["sample_id"].unique())
    print(f"Samples to plot: {sample_ids}")

    for sample_id in sample_ids:
        df_sample = df[df["sample_id"] == sample_id].copy()

        for factor in FACTOR_NAMES:
            plot_sample_factor(df_sample, sample_id, factor, OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()