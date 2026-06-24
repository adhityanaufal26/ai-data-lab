"""
Day 2: Data Cleaning — Example
==============================

Contoh penggunaan data cleaning pipeline pada data kotor.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_csv, quick_preview
from src.data.cleaner import clean_pipeline


def main():
    # Load data kotor
    print("📂 Loading messy data...\n")
    df = load_csv("data/raw/sample_kotor.csv")
    quick_preview(df)
    
    # Jalankan cleaning pipeline
    print("\n")
    df_clean = clean_pipeline(
        df,
        numeric_cols=["umur", "gaji"],
        text_cols=["nama", "kota", "email"],
        outlier_cols=["umur", "gaji"],
        fill_strategy="auto"
    )
    
    # Preview hasil
    print("\n📊 Cleaned data:")
    quick_preview(df_clean)
    
    # Simpan hasil bersih
    output_path = "data/raw/sample_bersih.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\n💾 Saved cleaned data to {output_path}")


if __name__ == "__main__":
    main()
