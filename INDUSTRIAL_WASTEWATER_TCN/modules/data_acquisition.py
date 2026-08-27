"""
Module 1 — Data Acquisition
============================

Responsibilities:
  • Load the wastewater CSV dataset.
  • Validate the dataset (required columns, non-empty).
  • Check for missing values and duplicate records.
  • Return a clean raw dataset for preprocessing.

The 5 wastewater parameters expected in the dataset:
    pH, COD_mg_L, BOD_mg_L, TDS_mg_L, Temperature_C
"""

import os
import pandas as pd


# Columns that MUST be present in the dataset
REQUIRED_COLUMNS = [
    "Timestamp",
    "pH",
    "COD_mg_L",
    "BOD_mg_L",
    "TDS_mg_L",
    "Temperature_C",
]


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the wastewater CSV file.

    Returns
    -------
    pd.DataFrame
        The raw loaded dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Loaded dataset from: {filepath}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Validate that the DataFrame contains all required columns
    and is not empty.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to validate.

    Returns
    -------
    bool
        True if the dataset passes all validation checks.

    Raises
    ------
    ValueError
        If required columns are missing or the dataset is empty.
    """
    if df.empty:
        raise ValueError("Dataset is empty (no rows or no columns).")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    print("Validation passed: all required columns are present.")
    print(f"Required columns: {REQUIRED_COLUMNS}")
    return True


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """
    Count and report missing (NaN) values per column.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect.

    Returns
    -------
    pd.Series
        Series indexed by column name with missing-value counts.
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame(
        {"missing_count": missing, "missing_pct": missing_pct}
    )
    print("\n--- Missing Values Report ---")
    print(missing_report.to_string())
    return missing


def check_duplicates(df: pd.DataFrame) -> int:
    """
    Count and report fully duplicated rows.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect.

    Returns
    -------
    int
        Number of duplicate rows.
    """
    dup_count = df.duplicated().sum()
    print(f"\n--- Duplicate Records ---")
    print(f"Duplicate rows found: {dup_count}")
    return dup_count


def check_chronological_order(df: pd.DataFrame) -> bool:
    """
    Check whether the Timestamp column is sorted chronologically.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to inspect.

    Returns
    -------
    bool
        True if timestamps are in chronological order.
    """
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    is_sorted = ts.is_monotonic_increasing
    print(f"\n--- Chronological Ordering ---")
    print(f"Timestamps sorted chronologically: {is_sorted}")
    if not is_sorted:
        print("  (Will be sorted during preprocessing)")
    return is_sorted


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove fully duplicated rows, keeping the first occurrence.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset with potential duplicates.

    Returns
    -------
    pd.DataFrame
        Dataset with duplicates removed.
    """
    before = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df_clean)
    print(f"Removed {removed} duplicate rows ({before} -> {len(df_clean)})")
    return df_clean


def acquire_and_validate(filepath: str, remove_dups: bool = False) -> pd.DataFrame:
    """
    End-to-end data acquisition pipeline:
    load, validate, report missing values and duplicates.

    Parameters
    ----------
    filepath : str
        Path to the CSV dataset.
    remove_dups : bool, default False
        If True, remove duplicate rows from the returned dataset.

    Returns
    -------
    pd.DataFrame
        The validated raw dataset (duplicates removed only if
        remove_dups=True).
    """
    df = load_dataset(filepath)
    validate_dataset(df)
    check_missing_values(df)
    dup_count = check_duplicates(df)
    check_chronological_order(df)

    if dup_count > 0 and remove_dups:
        df = remove_duplicates(df)

    print(f"\nData acquisition complete. Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    DATASET_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "dataset",
        "wastewater_10000.csv",
    )

    df = acquire_and_validate(DATASET_PATH, remove_dups=True)
    print("\n--- Dataset Preview ---")
    print(df.head())
    print("\n--- Dataset Info ---")
    print(df.info())
