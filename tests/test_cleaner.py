"""
Tests for Day 2: Data Cleaning
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cleaner import (
    clean_column_names, remove_empty_rows, remove_duplicates,
    fill_missing, fix_numeric_columns, standardize_text,
    remove_outliers_iqr, clean_pipeline
)


class TestCleanColumnNames:
    def test_strips_whitespace(self):
        df = pd.DataFrame({"  nama  ": [1], " umur ": [2]})
        result = clean_column_names(df)
        assert list(result.columns) == ["nama", "umur"]
    
    def test_lowercase(self):
        df = pd.DataFrame({"Nama": [1], "UMUR": [2]})
        result = clean_column_names(df)
        assert list(result.columns) == ["nama", "umur"]
    
    def test_spaces_to_underscores(self):
        df = pd.DataFrame({"nama lengkap": [1], "tanggal lahir": [2]})
        result = clean_column_names(df)
        assert list(result.columns) == ["nama_lengkap", "tanggal_lahir"]


class TestRemoveEmptyRows:
    def test_removes_all_nan_rows(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, None, 6]})
        result = remove_empty_rows(df)
        assert len(result) == 2
    
    def test_keeps_partial_nan(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        result = remove_empty_rows(df)
        assert len(result) == 3


class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result = remove_duplicates(df)
        assert len(result) == 2
    
    def test_subset_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 4, 4]})
        result = remove_duplicates(df, subset=["a"])
        assert len(result) == 2


class TestFillMissing:
    def test_auto_fill_numeric(self):
        df = pd.DataFrame({"x": [1, 2, None, 4]})
        result = fill_missing(df, strategy="auto")
        assert result["x"].isnull().sum() == 0
        assert result["x"].iloc[2] == 2.0  # median
    
    def test_auto_fill_text(self):
        df = pd.DataFrame({"x": ["a", None, "c"]})
        result = fill_missing(df, strategy="auto")
        assert result["x"].iloc[1] == "unknown"
    
    def test_custom_fill_values(self):
        df = pd.DataFrame({"x": [1, None, 3]})
        result = fill_missing(df, fill_values={"x": 99})
        assert result["x"].iloc[1] == 99


class TestFixNumericColumns:
    def test_converts_mixed_values(self):
        df = pd.DataFrame({"x": ["1", "abc", "3"]})
        result = fix_numeric_columns(df, ["x"])
        assert result["x"].dtype in [np.float64, float]
        assert pd.isna(result["x"].iloc[1])
    
    def test_handles_clean_numeric(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = fix_numeric_columns(df, ["x"])
        assert result["x"].dtype in [np.int64, np.float64, int, float]


class TestStandardizeText:
    def test_lowercase_and_strip(self):
        df = pd.DataFrame({"x": ["  Hello ", "WORLD  ", " foo "]})
        result = standardize_text(df, ["x"])
        assert list(result["x"]) == ["hello", "world", "foo"]


class TestRemoveOutliersIQR:
    def test_removes_extreme_values(self):
        df = pd.DataFrame({"x": [10, 11, 12, 13, 14, 100]})
        result = remove_outliers_iqr(df, "x")
        assert len(result) < len(df)
    
    def test_keeps_normal_values(self):
        df = pd.DataFrame({"x": [10, 11, 12, 13, 14]})
        result = remove_outliers_iqr(df, "x")
        assert len(result) == len(df)


class TestCleanPipeline:
    def test_full_pipeline(self):
        df = pd.DataFrame({
            "  Nama  ": ["Andi", "Budi", "Andi", None, "Eka"],
            " umur ": [25, None, 25, 28, 999],
            " Kota ": ["Jakarta", "bandung", "Jakarta", "Jakarta", "Bali"],
            " gaji ": [5000000, 8000000, 5000000, 7000000, -1000]
        })
        result = clean_pipeline(
            df,
            numeric_cols=["umur", "gaji"],
            text_cols=["nama", "kota"],
            outlier_cols=["gaji"]
        )
        # Should have clean columns
        assert "nama" in result.columns
        # Should have no duplicates
        assert len(result) == result.drop_duplicates().shape[0]
