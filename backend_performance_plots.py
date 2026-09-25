import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

# === CONFIG ===
project_path = Path("OKR_MICROBEADS_BASELINE-JAG-2025-04-30")
csv_path = project_path / "dlc-models-pytorch/iteration-0/OKR_MICROBEADS_BASELINEApr30-trainset95shuffle1/train/learning_stats.csv"

# Create output folder
output_dir = project_path / "training_performance_summaries"
output_dir.mkdir(parents=True, exist_ok=True)

# Generate unique filename if one already exists
base_name = "dlc_training_summary"
suffix = 0
while True:
    candidate = output_dir / f"{base_name}_{suffix}.pdf" if suffix else output_dir / f"{base_name}.pdf"
    if not candidate.exists():
        output_pdf = candidate
        break
    suffix += 1


# Load data
df = pd.read_csv(csv_path)

# Define dense columns (recorded every step)
dense_cols = [
    'losses/train.bodypart_heatmap',
    'losses/train.bodypart_locref',
    'losses/train.bodypart_total_loss',
    'losses/train.total_loss'
]

# Define sparse columns (recorded occasionally)
sparse_cols = [col for col in df.columns if col not in dense_cols + ['step']]

# === Plot and Save to PDF ===
with PdfPages(output_pdf) as pdf:
    # Plot dense metrics
    for col in dense_cols:
        plt.figure(figsize=(10, 4))
        plt.plot(df['step'], df[col], label=col)
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title(f"{col} (Dense)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Plot sparse metrics (drop NaNs)
    for col in sparse_cols:
        if df[col].notna().sum() == 0:
            continue  # skip completely empty columns
        plt.figure(figsize=(10, 4))
        plt.plot(df['step'][df[col].notna()], df[col].dropna(), label=col, marker='o')
        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title(f"{col} (Sparse)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

print(f"\n✅ All plots saved to PDF: {output_pdf}")



