"""
Day 1: Data Loading
===================

Belajar cara load data dari berbagai sumber:
- CSV file
- JSON file
- URL/API

Concepts yang dipelajari:
- pandas DataFrame
- Error handling
- Data validation
"""

import pandas as pd
import json
from pathlib import Path
from typing import Optional


def load_csv(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load data dari CSV file.
    
    Args:
        filepath: Path ke CSV file
        **kwargs: Parameter tambahan untuk pd.read_csv()
        
    Returns:
        pd.DataFrame: Data yang sudah di-load
        
    Raises:
        FileNotFoundError: Kalau file tidak ditemukan
        ValueError: Kalau file kosong atau format salah
        
    Example:
        >>> df = load_csv("data/raw/sales.csv")
        >>> print(df.shape)
        (1000, 5)
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")
    
    if not path.suffix.lower() == '.csv':
        raise ValueError(f"File bukan CSV: {path.suffix}")
    
    df = pd.read_csv(filepath, **kwargs)
    
    if df.empty:
        raise ValueError(f"File kosong: {filepath}")
    
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns from {path.name}")
    return df


def load_json(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load data dari JSON file.
    
    Args:
        filepath: Path ke JSON file
        **kwargs: Parameter tambahan untuk pd.read_json()
        
    Returns:
        pd.DataFrame: Data yang sudah di-load
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {filepath}")
    
    df = pd.read_json(filepath, **kwargs)
    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns from {path.name}")
    return df


def load_from_url(url: str, fmt: str = "csv", **kwargs) -> pd.DataFrame:
    """
    Load data dari URL.
    
    Args:
        url: URL data source
        fmt: Format data ('csv', 'json', 'excel')
        **kwargs: Parameter tambahan untuk pd.read_*()
        
    Returns:
        pd.DataFrame: Data yang sudah di-load
        
    Example:
        >>> url = "https://raw.githubusercontent.com/datasets/..." 
        >>> df = load_from_url(url, fmt="csv")
    """
    loaders = {
        "csv": pd.read_csv,
        "json": pd.read_json,
        "excel": pd.read_excel,
    }
    
    if fmt not in loaders:
        raise ValueError(f"Format tidak support: {fmt}. Gunakan: {list(loaders.keys())}")
    
    df = loaders[fmt](url, **kwargs)
    print(f"✅ Loaded {len(df)} rows from URL ({fmt})")
    return df


def quick_preview(df: pd.DataFrame, n: int = 5) -> None:
    """
    Quick preview DataFrame — shape, dtypes, sample rows.
    
    Args:
        df: DataFrame
        n: Jumlah sample rows
    """
    print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n📋 Columns:")
    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        print(f"  {col}: {dtype} ({nulls} nulls)")
    print(f"\n🔍 Sample ({n} rows):")
    print(df.head(n).to_string())
