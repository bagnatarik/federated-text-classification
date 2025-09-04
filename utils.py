import os, re

import pandas as pd

from datasets import load_dataset

# === REGULAR EXPRESSIONS ===
RE_URL   = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w{2,}\b")
RE_HTML  = re.compile(r"<[^>]+>")

# === FUNCTIONS ===

def make_directory(path: str, sub_path: str | None = None) -> None :
    """
        Create a directory if it doesn't exist.

        Args:
            path: The main directory.
            sub_path: The sub directory.
    """
    if not sub_path and not os.path.exists(path):
        os.makedirs(path)
        return print(f"'{path}' Directory created.")
    elif not sub_path and os.path.exists(path):
        return print(f"'{path}' Directory already created.")
    
    directory = os.path.join(path, sub_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
        return print(f"'{directory}' Directory created.")
    
    return print(f"'{directory}' Directory already created.")

def load_enron_dataset() -> pd.DataFrame:
    """
        Load the SetFit Enron Spam dataset from Hugginface.

        return: The SetFit Enron Spam dataset as a pandas DataFrame.
    """
    dataset = load_dataset("SetFit/enron_spam")

    # This dataset contains trainset and test set we need to concat them
    splits: list[str] = [key for key in dataset.keys()]
    dataframes: list[pd.DataFrame] = [dataset[key].to_pandas() for key in splits]

    return pd.concat(dataframes, ignore_index=True)

def parse_to_csv(dataframe: pd.DataFrame, csv_name: str, path: str) -> None:
    """
        Parse the given DataFrame to a CSV file.

        Args:
            dataframe: The DataFrame to be parsed.
            csv_name: The name of the CSV file.
            path: The path to the CSV file.
    """

    dataframe.to_csv(f"{path}\\{csv_name}", index=False)
    return print(f"Dataset parsed to {path}\\{csv_name}.")