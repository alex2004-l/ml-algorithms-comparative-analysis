from labelsDict import CONTINUE, DISCRETE, LABEL, HeartDisease
from utils import plot_boxplot_value_range, plot_description_values_table, plot_barplot_features, plot_correlation_matrix, plot_chi_pvals_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import chi2_contingency
import itertools

HEART_DATASET_PATH_TRAIN    = 'datasets/heart_4_train.csv'
HEART_DATASET_PATH_TEST     = 'datasets/heart_4_test.csv'
HEART_CONTINUE_VARS_TABLE   = 'tables/heart_continuous.png'
HEART_DISCRETE_VARS_TABLE   = 'tables/heart_discrete.png'
HEART_CONTINUE_VARS_BOXPLOT = 'plots/boxplot_continuous_heart.png'
HEART_DISCRETE_VARS_BARPLOT = 'plots/barplot_discrete_heart.png'
HEART_LABEL_VARS_BARPLOT    = 'plots/barplot_label_heart.png'
HEART_CORRELATION_MATRIX    = 'correlation/heart_correlation.png'

def heart_eda_statistics():
    if not os.path.exists(os.path.join(os.getcwd(), 'datasets', 'heart_combined.csv')):
        df_train = pd.read_csv(HEART_DATASET_PATH_TRAIN)
        df_test = pd.read_csv(HEART_DATASET_PATH_TEST)

        dataset = pd.concat([df_train, df_test], ignore_index=True)
    else:
        dataset = pd.read_csv(os.path.join(os.getcwd(), 'datasets', 'heart_combined.csv'))

    __get_eda_feature_statistics(dataset.copy())
    __get_eda_label_statistics(dataset.copy())
    __get_eda_correlation_statistics(dataset.copy())

def __get_eda_feature_statistics(dataset: pd.DataFrame):
    continuous_values = [col for col in dataset if HeartDisease[col]==CONTINUE]
    discrete_values = [col for col in dataset if HeartDisease[col]==DISCRETE]

    # Table for continuous variables
    plot_description_values_table(dataset[continuous_values].describe().T.reset_index(),title="Heart Disease Continuous Variables", outputname=HEART_CONTINUE_VARS_TABLE, figsize=(14, 5))

    # Boxplot for continuous variables
    plot_boxplot_value_range(dataset[continuous_values], title="Heart Disease Boxplot", outputname=HEART_CONTINUE_VARS_BOXPLOT, figsize=(14, 10), log_scale=True)

    df_discrete_vals = pd.DataFrame({
        'index': discrete_values,
        'count': [dataset[col].count() for col in discrete_values],
        'no_unique_values': [dataset[col].nunique() for col in discrete_values],
        'value_counts': [dataset[col].value_counts().to_dict() for col in discrete_values]
    })

    plot_description_values_table(df_discrete_vals, title="Heart Disease Discrete Variables", outputname=HEART_DISCRETE_VARS_TABLE, figsize=(14, 5))

    # Table for discrete variables
    plot_barplot_features(dataset[discrete_values], values=discrete_values, outputname=HEART_DISCRETE_VARS_BARPLOT)


def __get_eda_label_statistics(dataset: pd.DataFrame):
    label_values = [col for col in dataset if HeartDisease[col]==LABEL]

    plot_barplot_features(dataset[label_values], label_values, HEART_LABEL_VARS_BARPLOT)


def __get_eda_correlation_statistics(dataset: pd.DataFrame):
    columns = [col for col in dataset if HeartDisease[col] in [CONTINUE, DISCRETE]]
    correlation_matrix = dataset[columns].corr(method='pearson')

    plot_correlation_matrix(correlation_matrix, HEART_CORRELATION_MATRIX)
    df = chi_square_all_pairs(dataset, [col for col in dataset if HeartDisease[col] in [DISCRETE]], alpha=0.05)
    plot_chi_pvals_matrix(df, outputname='chi_square_results_heart1.png', figsize=(10, 8), value='p-value')
    plot_chi_pvals_matrix(df, outputname='chi_square_results_heart2.png', figsize=(10, 8), value='Independent?')
    plt.close()

def chi_square_all_pairs(df, categorical_cols, alpha=0.05):
    results = []
    for col1, col2 in itertools.permutations(categorical_cols, 2):
        contingency = pd.crosstab(df[col1], df[col2])
        chi2, p, dof, expected = chi2_contingency(contingency)
        result = {
            'Var1': col1,
            'Var2': col2,
            'Chi2': round(chi2, 4),
            'p-value': round(p, 4),
            'dof': dof,
            'Independent?': 1 if p > alpha else 0
        }
        results.append(result)
    return pd.DataFrame(results)