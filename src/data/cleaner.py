"""
Day 2: Data Cleaning
====================

Belajar cara bersihin data kotor:
- Handle missing values (NaN/null)
- Remove duplicates
- Fix data types
- Handle outliers
- Standardize text (lowercase, strip whitespace)
- Validate data ranges

Concepts yang dipelajari:
- pandas .isnull(), .dropna(), .fillna()
- .duplicated(), .drop_duplicates()
- .astype(), pd.to_numeric()
- .str.lower(), .str.strip()
- IQR outlier detection
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standarisasi nama kolom: lowercase, strip whitespace, ganti spasi dengan underscore.
    
    Args:
        df: DataFrame
        
    Returns:
        DataFrame dengan nama kolom yang sudah dibersihkan
    """
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_')
    )
    print(f"✅ Cleaned column names: {list(df.columns)}")
    return df


def remove_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hapus baris yang semua kolomnya kosong/NaN.
    
    Args:
        df: DataFrame
        
    Returns:
        DataFrame tanpa baris kosong
    """
    before = len(df)
    df = df.dropna(how='all')
    removed = before - len(df)
    if removed > 0:
        print(f"✅ Removed {removed} completely empty rows")
    return df


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Hapus baris duplikat.
    
    Args:
        df: DataFrame
        subset: Kolom yang dijadikan basis pengecekan duplikat.
                None = cek semua kolom.
    
    Returns:
        DataFrame tanpa duplikat
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep='first')
    removed = before - len(df)
    if removed > 0:
        label = f" (based on {subset})" if subset else ""
        print(f"✅ Removed {removed} duplicate rows{label}")
    return df


def fill_missing(df: pd.DataFrame, strategy: str = "auto", 
                 fill_values: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    Isi missing values berdasarkan strategi.
    
    Args:
        df: DataFrame
        strategy: 'auto' (smart fill), 'mean', 'median', 'zero', 'drop'
        fill_values: Dict {kolom: nilai} untuk manual fill
    
    Returns:
        DataFrame dengan missing values yang sudah diisi
        
    Strategy 'auto':
        - Numeric → median
        - Text → 'unknown'
        - Boolean → mode
    """
    df = df.copy()
    
    if fill_values:
        for col, val in fill_values.items():
            if col in df.columns:
                filled = df[col].isnull().sum()
                df[col] = df[col].fillna(val)
                if filled > 0:
                    print(f"  Filled {filled} nulls in '{col}' with {val}")
        return df
    
    if strategy == "drop":
        before = len(df)
        df = df.dropna()
        print(f"✅ Dropped {before - len(df)} rows with any null")
        return df
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue
            
        if strategy in ("mean", "median", "zero"):
            if pd.api.types.is_numeric_dtype(df[col]):
                if strategy == "mean":
                    fill = df[col].mean()
                elif strategy == "median":
                    fill = df[col].median()
                else:
                    fill = 0
                df[col] = df[col].fillna(fill)
                print(f"  Filled {null_count} nulls in '{col}' → {strategy} ({fill:.2f})")
        elif strategy == "auto":
            if pd.api.types.is_numeric_dtype(df[col]):
                fill = df[col].median()
                df[col] = df[col].fillna(fill)
                print(f"  Filled {null_count} nulls in '{col}' → median ({fill:.2f})")
            else:
                df[col] = df[col].fillna("unknown")
                print(f"  Filled {null_count} nulls in '{col}' → 'unknown'")
    
    return df


def fix_numeric_columns(df: pd.DataFrame, columns: List[str], 
                         coerce: bool = True) -> pd.DataFrame:
    """
    Konversi kolom ke numeric. Non-numeric values jadi NaN.
    
    Args:
        df: DataFrame
        columns: List kolom yang mau di-fix
        coerce: True = paksa jadi numeric (errors → NaN)
    
    Returns:
        DataFrame dengan tipe data yang sudah diperbaiki
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            original_dtype = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce' if coerce else 'raise')
            nulls = df[col].isnull().sum()
            print(f"  '{col}': {original_dtype} → {df[col].dtype}" + 
                  f" ({nulls} values became NaN)" if nulls > 0 else 
                  f"  '{col}': {original_dtype} → {df[col].dtype}")
    return df


def standardize_text(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Standarisasi kolom text: lowercase + strip whitespace.
    
    Args:
        df: DataFrame
        columns: Kolom yang mau di-standarisasi
    
    Returns:
        DataFrame dengan text yang sudah distandarisasi
    """
    df = df.copy()
    for col in columns:
        if col in df.columns and pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip().str.lower()
            print(f"  Standardized '{col}' → lowercase + stripped")
    return df


def remove_outliers_iqr(df: pd.DataFrame, column: str, 
                         multiplier: float = 1.5) -> pd.DataFrame:
    """
    Hapus outliers berdasarkan IQR method.
    
    Args:
        df: DataFrame
        column: Kolom numeric yang mau di-check
        multiplier: IQR multiplier (default 1.5, gunakan 3.0 untuk extreme outliers)
    
    Returns:
        DataFrame tanpa outliers
    """
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        print(f"  ⚠️ '{column}' bukan numeric, skip")
        return df
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR
    
    before = len(df)
    df = df[(df[column] >= lower) & (df[column] <= upper)]
    removed = before - len(df)
    
    if removed > 0:
        print(f"  Removed {removed} outliers from '{column}' " +
              f"(range: {lower:.2f} - {upper:.2f})")
    return df


def clean_pipeline(df: pd.DataFrame, 
                   numeric_cols: Optional[List[str]] = None,
                   text_cols: Optional[List[str]] = None,
                   outlier_cols: Optional[List[str]] = None,
                   fill_strategy: str = "auto") -> pd.DataFrame:
    """
    Full cleaning pipeline — jalankan semua step sekaligus.
    
    Args:
        df: DataFrame kotor
        numeric_cols: Kolom yang harusnya numeric
        text_cols: Kolom text yang perlu distandarisasi
        outlier_cols: Kolom yang perlu outlier removal
        fill_strategy: Strategi fill missing values
    
    Returns:
        DataFrame yang sudah bersih
        
    Example:
        >>> df_clean = clean_pipeline(
        ...     df_kotor,
        ...     numeric_cols=['umur', 'gaji'],
        ...     text_cols=['nama', 'kota'],
        ...     outlier_cols=['gaji']
        ... )
    """
    print("=" * 50)
    print("🧹 Starting Data Cleaning Pipeline")
    print("=" * 50)
    print(f"\n📊 Initial: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Step 1: Clean column names
    print("\n📌 Step 1: Clean column names")
    df = clean_column_names(df)
    
    # Update col references to lowercase
    if numeric_cols:
        numeric_cols = [c.lower().strip().replace(' ', '_') for c in numeric_cols]
    if text_cols:
        text_cols = [c.lower().strip().replace(' ', '_') for c in text_cols]
    if outlier_cols:
        outlier_cols = [c.lower().strip().replace(' ', '_') for c in outlier_cols]
    
    # Step 2: Remove empty rows
    print("\n📌 Step 2: Remove empty rows")
    df = remove_empty_rows(df)
    
    # Step 3: Standardize text
    if text_cols:
        print("\n📌 Step 3: Standardize text")
        df = standardize_text(df, text_cols)
    
    # Step 4: Fix numeric columns
    if numeric_cols:
        print("\n📌 Step 4: Fix numeric columns")
        df = fix_numeric_columns(df, numeric_cols)
    
    # Step 5: Remove duplicates
    print("\n📌 Step 5: Remove duplicates")
    df = remove_duplicates(df)
    
    # Step 6: Fill missing values
    print(f"\n📌 Step 6: Fill missing values (strategy: {fill_strategy})")
    df = fill_missing(df, strategy=fill_strategy)
    
    # Step 7: Remove outliers
    if outlier_cols:
        print("\n📌 Step 7: Remove outliers")
        for col in outlier_cols:
            df = remove_outliers_iqr(df, col)
    
    print(f"\n✅ Final: {df.shape[0]} rows × {df.shape[1]} columns")
    print("=" * 50)
    
    return df.reset_index(drop=True)
