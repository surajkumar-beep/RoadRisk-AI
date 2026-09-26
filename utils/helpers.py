import pandas as pd

from utils.logger import logger


def load_dataset(path):
    """Load a dataset from the given path."""
    logger.info(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    logger.info(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df


def display_shape(df):
    """Display the shape of the dataframe."""
    shape = df.shape
    logger.info(f"DataFrame shape: {shape[0]} rows x {shape[1]} columns")
    return shape


def display_columns(df):
    """Display the column names of the dataframe."""
    columns = list(df.columns)
    logger.info(f"DataFrame columns: {columns}")
    return columns