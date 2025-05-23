from labelsDict import CONTINUE, DISCRETE, LABEL, UNLABELED, Pirvision
from utils import plot_boxplot_value_range, plot_description_values_table, plot_barplot_features, plot_correlation_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

PIRVISION_DATASET_PATH_TRAIN    = 'datasets/pirvision_office_train.csv'
PIRVISION_DATASET_PATH_TEST     = 'datasets/pirvision_office_test.csv'
PIRVISION_CONTINUE_VARS_TABLE   = 'tables/pirvision_continuous.png'
PIRVISION_DISCRETE_VARS_TABLE   = 'tables/pirvision_discrete.png'
PIRVISION_CONTINUE_VARS_BOXPLOT = 'plots/boxplot_continuous_pirvision.png'
PIRVISION_DISCRETE_VARS_BARPLOT = 'plots/barplot_discrete_pirvision.png'
PIRVISION_LABEL_VARS_BARPLOT    = 'plots/barplot_label_pirvision.png'
PIRVISION_CORRELATION_MATRIX    = 'correlation/pirvision_correlation.png'

def pirvision_eda_statistics():
    if not os.path.exists(os.path.join(os.getcwd(), 'datasets', 'pirvision_combined.csv')):
        df_train = pd.read_csv(PIRVISION_DATASET_PATH_TRAIN)
        df_test = pd.read_csv(PIRVISION_DATASET_PATH_TEST)

        dataset = pd.concat([df_train, df_test], ignore_index=True)
    else:
        dataset = pd.read_csv(os.path.join(os.getcwd(), 'datasets', 'pirvision_combined.csv'))

    __get_eda_feature_statistics(dataset.copy())
    __get_eda_label_statistics(dataset.copy())
    __get_eda_correlation_statistics(dataset.copy())


def __get_eda_feature_statistics(dataset: pd.DataFrame):
    continuous_values = [col for col in dataset if Pirvision[col]==CONTINUE]
    discrete_values = [col for col in dataset if Pirvision[col]==DISCRETE]

    df_discrete_vals = pd.DataFrame({
        'index': discrete_values,
        'count': [dataset[col].count() for col in discrete_values],
        'no_unique_values': [dataset[col].nunique() for col in discrete_values],
        'value_counts': [dataset[col].value_counts().to_dict() for col in discrete_values]
    })

    plot_description_values_table(df_discrete_vals, title="Pirvision Discrete Variables", outputname=PIRVISION_DISCRETE_VARS_TABLE, figsize=(14, 2))
    plot_description_values_table(dataset[continuous_values].describe().T.reset_index(),title="Pirvision Continuous Variables", outputname=PIRVISION_CONTINUE_VARS_TABLE, figsize=(15, 24))

    # Boxplot only for OBS_1
    plt.figure(figsize=(5, 5))
    sns.boxplot(x=dataset['OBS_1'], color='lightblue')
    plt.title('Boxplot of OBS_1')
    plt.xlabel('Value')
    plt.savefig(PIRVISION_CONTINUE_VARS_BOXPLOT.replace('.png', f'_OBS_1.png'), dpi=300)

    # Boxplot for temperatures
    plot_boxplot_value_range(dataset[['Temp (F)', 'Temp (C)']], title = f"Pirvision Boxplot Continuous Variables Temp", outputname=PIRVISION_CONTINUE_VARS_BOXPLOT.replace('.png', f'_Temp.png'), figsize=(6, 6), log_scale=False)

    continuous_values.remove('OBS_1')
    continuous_values.remove('Temp (F)')
    continuous_values.remove('Temp (C)')

    step = 14
    iterations = 4

    for i in range(iterations):
        continuous_vals = continuous_values[i* step:(i+1) * step]
        plot_boxplot_value_range(dataset[continuous_vals], title = f"Pirvision Boxplot Continuous Variables {i + 1}", outputname=PIRVISION_CONTINUE_VARS_BOXPLOT.replace('.png', f'_{i}.png'), figsize=(16, 9), log_scale=False)

    plot_barplot_features(dataset, discrete_values, PIRVISION_DISCRETE_VARS_BARPLOT)

def __get_eda_label_statistics(dataset: pd.DataFrame):
    label_vals = [col for col in dataset if Pirvision[col]==LABEL]

    plot_barplot_features(dataset, label_vals, PIRVISION_LABEL_VARS_BARPLOT)

def __get_eda_correlation_statistics(dataset: pd.DataFrame):
    columns = [col for col in dataset if not Pirvision[col]==UNLABELED and not Pirvision[col]==LABEL]
    correlation_matrix = dataset[columns].corr(method='pearson')

    plot_correlation_matrix(correlation_matrix, PIRVISION_CORRELATION_MATRIX, figsize=(25, 25))