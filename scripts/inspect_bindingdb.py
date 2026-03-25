#!/usr/bin/env python3
"""
Inspect BindingDB dataset from cached pickle file.
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "BindingDB.pkl"

if __name__ == "__main__":
    df = pd.read_pickle(DATA_PATH)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())
