from labelsDict import CONTINUE, DISCRETE, Pirvision
from utils import plot_boxplot_value_range, plot_description_values_table
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
PIRVISION_DISCRETE_VARS_BOXPLOT = 'plots/boxplot_discrete_pirvision.png'

def solve_first_eda_pirvision():
    if not os.path.exists(os.path.join(os.getcwd(), 'datasets', 'pirvision_combined.csv')):
        df_train = pd.read_csv(PIRVISION_DATASET_PATH_TRAIN)
        df_test = pd.read_csv(PIRVISION_DATASET_PATH_TEST)

        dataset = pd.concat([df_train, df_test], ignore_index=True)
    else:
        dataset = pd.read_csv(os.path.join(os.getcwd(), 'datasets', 'pirvision_combined.csv'))

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

    plt.figure(figsize=(5, 5))
    sns.boxplot(x=dataset['OBS_1'], color='lightblue')
    
    plt.title('Boxplot of OBS_1')
    plt.xlabel('Value')
    plt.savefig(PIRVISION_CONTINUE_VARS_BOXPLOT.replace('.png', f'_OBS_1.png'), dpi=300)

    # (pd.DataFrame(dataset['OBS_1']), title = f"Pirvision Boxplot Continuous Variables OBS_1", outputname=PIRVISION_CONTINUE_VARS_BOXPLOT.replace('.png', f'_OBS_1.png'), figsize=(16, 9), log_scale=False)
    plot_boxplot_value_range(dataset[['Temp (F)', 'Temp (C)']], title = f"Pirvision Boxplot Continuous Variables Temp", outputname=PIRVISION_CONTINUE_VARS_BOXPLOT.replace('.png', f'_Temp.png'), figsize=(6, 6), log_scale=False)
    continuous_values.remove('OBS_1')
    continuous_values.remove('Temp (F)')
    continuous_values.remove('Temp (C)')

    for i in range(4):
        continuous_vals = continuous_values[i*14:(i+1)*14]
        plot_boxplot_value_range(dataset[continuous_vals], title = f"Pirvision Boxplot Continuous Variables {i}", outputname=PIRVISION_CONTINUE_VARS_BOXPLOT.replace('.png', f'_{i}.png'), figsize=(16, 9), log_scale=False)

    for value in discrete_values:
        plt.figure(figsize=(4, 4))
        dataset[value].value_counts().sort_index().plot.bar(width=0.3, color=['skyblue', 'lightcoral', 'forestgreen', 'magenta'], edgecolor='black')
        plt.xlabel(value, fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        plt.savefig(PIRVISION_DISCRETE_VARS_BOXPLOT.replace('.png', f'_{value}.png'), bbox_inches='tight', dpi=300)