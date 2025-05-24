import pandas as pd
import os

PIRVISION_DATASET_PATH_TRAIN    = 'datasets/pirvision_office_train.csv'
PIRVISION_DATASET_PATH_TEST     = 'datasets/pirvision_office_test.csv'

def pirvision_preprocessing():
    df_train = pd.read_csv(PIRVISION_DATASET_PATH_TRAIN)
    df_test = pd.read_csv(PIRVISION_DATASET_PATH_TEST)

    dataset = pd.concat([df_train, df_test], ignore_index=True)

    missing_values = df_train.isnull().sum()
    print(f"Missing values in training dataset:\n{missing_values[missing_values > 0]}")
    columns = [col for col in missing_values[missing_values > 0]]

    print(f"Columns with missing values: {columns}")
