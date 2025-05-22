from labelsDict import CONTINUE, DISCRETE, HeartDisease
from utils import plot_boxplot_value_range, plot_description_values_table
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

HEART_DATASET_PATH_TRAIN    = 'datasets/heart_4_train.csv'
HEART_DATASET_PATH_TEST     = 'datasets/heart_4_test.csv'
HEART_CONTINUE_VARS_TABLE   = 'tables/heart_continuous.png'
HEART_DISCRETE_VARS_TABLE   = 'tables/heart_discrete.png'
HEART_CONTINUE_VARS_BOXPLOT = 'plots/boxplot_continuous_heart.png'
HEART_DISCRETE_VARS_BOXPLOT = 'plots/boxplot_discrete_heart.png'

def solve_first_eda_heart():
    if not os.path.exists(os.path.join(os.getcwd(), 'datasets', 'heart_combined.csv')):
        df_train = pd.read_csv(HEART_DATASET_PATH_TRAIN)
        df_test = pd.read_csv(HEART_DATASET_PATH_TEST)

        dataset = pd.concat([df_train, df_test], ignore_index=True)
    else:
        dataset = pd.read_csv(os.path.join(os.getcwd(), 'datasets', 'heart_combined.csv'))

    continue_values = [col for col in dataset if HeartDisease[col]==CONTINUE]
    discrete_values = [col for col in dataset if HeartDisease[col]==DISCRETE]

    plot_description_values_table(dataset[continue_values].describe().T.reset_index(), HEART_CONTINUE_VARS_TABLE, figsize=(14, 5))

    df_discrete_vals = pd.DataFrame({
        'index': discrete_values,
        'count': [dataset[col].count() for col in discrete_values],
        'no_unique_values': [dataset[col].nunique() for col in discrete_values],
        'value_counts': [dataset[col].value_counts().to_dict() for col in discrete_values]
    })

    plot_description_values_table(df_discrete_vals, HEART_DISCRETE_VARS_TABLE, figsize=(14, 5))

    plot_boxplot_value_range(dataset[continue_values], HEART_CONTINUE_VARS_BOXPLOT, figsize=(10, 9), log_scale=True)

