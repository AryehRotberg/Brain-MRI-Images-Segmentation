import pandas as pd


def save_as_csv(
    dataframe: pd.DataFrame,
    output_path: str
) -> None:
    """
    Saves Pandas DataFrame to a CSV file, excluding index column.

    Args:
        dataframe (pd.DataFrame): DataFrame to be saved as a CSV file.
        output_path (str): File path where the CSV file will be saved.
    """
    dataframe.to_csv(output_path, index=False)


def load_from_csv(data_path: str) -> None:
    """
    Loads a CSV file into Pandas Dataframe.

    Args:
        data_path (str): Path where the CSV file is stored.
    """
    return pd.read_csv(data_path)
