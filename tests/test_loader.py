"""
Tests for data loader module.
"""

import pytest
import pandas as pd
import tempfile
import os
from src.data.loader import load_csv, quick_preview


class TestLoadCSV:
    """Test cases untuk load_csv function."""
    
    def test_load_valid_csv(self, tmp_path):
        """Test load CSV yang valid."""
        # Create temp CSV
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("name,age\nAlice,25\nBob,30\n")
        
        df = load_csv(str(csv_file))
        assert len(df) == 2
        assert list(df.columns) == ["name", "age"]
    
    def test_file_not_found(self):
        """Test error kalau file tidak ada."""
        with pytest.raises(FileNotFoundError):
            load_csv("nonexistent.csv")
    
    def test_wrong_format(self, tmp_path):
        """Test error kalau bukan CSV."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("not a csv")
        
        with pytest.raises(ValueError):
            load_csv(str(txt_file))


class TestQuickPreview:
    """Test cases untuk quick_preview function."""
    
    def test_preview_output(self, capsys):
        """Test preview menghasilkan output."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        quick_preview(df)
        
        captured = capsys.readouterr()
        assert "2 rows" in captured.out
        assert "2 columns" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
