"""
Day 1 Example: Load sample data dan explore
============================================

Run: python examples/day1_data_loading.py
"""

import sys
sys.path.insert(0, ".")

from src.data.loader import load_csv, load_from_url, quick_preview
import pandas as pd

# === Example 1: Create sample data ===
print("=" * 50)
print("📦 Example 1: Create & Load CSV")
print("=" * 50)

# Bikin sample data
sample_data = {
    "nama": ["Andi", "Budi", "Citra", "Dewi", "Eka"],
    "umur": [25, 30, 22, 28, 35],
    "kota": ["Jakarta", "Bandung", "Surabaya", "Jakarta", "Bali"],
    "gaji": [5000000, 8000000, 4500000, 7000000, 12000000],
}

df = pd.DataFrame(sample_data)
df.to_csv("data/raw/sample_karyawan.csv", index=False)

# Load balik
df_loaded = load_csv("data/raw/sample_karyawan.csv")
quick_preview(df_loaded)

# === Example 2: Load from URL ===
print("\n" + "=" * 50)
print("🌐 Example 2: Load from URL")
print("=" * 50)

# Titanic dataset — classic!
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
try:
    df_titanic = load_from_url(url, fmt="csv")
    quick_preview(df_titanic, n=3)
    
    # Save ke local
    df_titanic.to_csv("data/raw/titanic.csv", index=False)
    print("\n💾 Saved ke data/raw/titanic.csv")
except Exception as e:
    print(f"❌ Error: {e}")

# === Example 3: Basic exploration ===
print("\n" + "=" * 50)
print("🔍 Example 3: Basic Data Exploration")
print("=" * 50)

df = load_csv("data/raw/sample_karyawan.csv")

# Basic stats
print(f"\n📊 Rata-rata gaji: Rp {df['gaji'].mean():,.0f}")
print(f"📊 Umur termuda: {df['umur'].min()} tahun")
print(f"📊 Umur tertua: {df['umur'].max()} tahun")
print(f"📊 Jumlah kota unik: {df['kota'].nunique()}")

# Group by
print(f"\n📊 Gaji per kota:")
print(df.groupby("kota")["gaji"].mean().to_string())

print("\n✅ Day 1 selesai! Besok: Data Cleaning 🧹")
