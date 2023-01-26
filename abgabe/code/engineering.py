"""
Module performs data preprocessing and feature engineering.
"""
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def add_ageincome(df_edit: pd.DataFrame) -> pd.DataFrame:
    """
    creates column 'ageincome',
    calculated by dividing column "age" by the average of column "annual_income".

    :param pd.DataFrame df_edit: dataframe 
    :return: the edited dataframe
    """
    df_edit["ageincome"] = df_edit["Age"] / df_edit["annual_income"].mean()
    return df_edit


def rename_column(df_edit: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    """
    renames column 'old_name' to 'new_name'

    :param pd.DataFrame df_edit: dataframe
    :param list[str] column_names: old column name, new column name
    :return: the edited dataframe
    """

    for names in column_names:
        df_edit = df_edit.rename(columns={names[0]: names[1]})
    return df_edit


def build_df(path: str) -> pd.DataFrame:
    """
    builds data frame with all feature engineering methods, renames columns, creates new column

    :param tupel[list[str]] paths: list containing the path to all csv files
    :return: df containing the data of all csv-files
    """

    name_changes = [("Annual Income (k$)", "annual_income"), ("Spending Score (1-100)", "spending_score")]

    return normalize(add_ageincome(
        rename_column(pd.read_csv(path), name_changes).drop(["CustomerID"], axis=1))
            )

def convert_column_to_num(column: pd.Series) -> pd.Series:
    """
    converts values of column to numerical values using integers encoding method

    :param pd.Series column: column to be converted
    :return: converted column
    """
    return column.replace(column.unique(), range(1, len(column.unique()) + 1))



def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    normalizes data in order to get a smaller scale of values, utilizes numpy

    :param df: the dataframe to be normalized
    :return: the normalized dataframe
    """
    df_norm = df.copy()
    scaler = MinMaxScaler()
    df_norm[["ageincome", "spending_score"]] = scaler.fit_transform(df_norm[["ageincome", "spending_score"]])

    return df_norm


def main():
    csv_path = 'data\Mall_Customers.csv'

    df = build_df(csv_path) #build new csv
    os.makedirs('data/out', exist_ok=True)  # create out directory
    df.to_csv('data/out/clean_mall_customer.csv', index=False)  # put df in folder out


if __name__ == "__main__":
    main()
