import pandas as pd
from pandas import DataFrame

from backend.src.core.processor.base_processor import BaseProcessor


class CSVProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()

    @staticmethod
    def load_csv(self, csv_path: str, delimiter: str = ",", encoding: str = "utf-8") -> DataFrame:
        try:
            df = pd.read_csv(csv_path, delimiter=delimiter, encoding=encoding)
        except FileNotFoundError:
            raise ValueError(f"The file {self.file_path} was not found.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file {self.file_path} is empty.")
        return df

    @staticmethod
    def save_csv(self, df: DataFrame, save_path: str, delimiter: str = ",", encoding: str = "utf-8"):
        df.to_csv(save_path, sep=delimiter, encoding=encoding, index=False)
        return save_path

    @staticmethod
    def preview(self, df: DataFrame, n: int = 5):
        return df.head(n)
