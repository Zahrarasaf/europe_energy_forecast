import os
import pandas as pd
import zipfile


def load_csv_from_zip(zip_path: str, target_filename: str = None) -> pd.DataFrame:
    """
    Load a CSV file from a ZIP archive.
    If target_filename is None, loads the first CSV found inside the ZIP file.
    """
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as z:
        file_list = z.namelist()

        # Auto-detect CSV if not provided
        if target_filename is None:
            csv_files = [f for f in file_list if f.endswith(".csv")]
            if len(csv_files) == 0:
                raise ValueError("No CSV file found inside the ZIP archive.")
            target_filename = csv_files[0]

        with z.open(target_filename) as f:
            df = pd.read_csv(f)

    return df


def load_csv(path: str) -> pd.DataFrame:
    """Directly loads a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def save_processed(df: pd.DataFrame, output_path: str):
    """Save cleaned/processed dataframe."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
