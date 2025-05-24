import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import table
import numpy as np
import itertools
from scipy.stats import chi2_contingency

def plot_boxplot_value_range(dataset: pd.DataFrame, title:str=None, outputname: str = None, figsize: tuple = (10, 6), log_scale: bool = False):
    plt.figure(figsize=figsize)

    sns.set(style="whitegrid")
    melted_df = dataset.melt(var_name="Feature", value_name="Value")

    ax = sns.boxplot(x="Feature", y="Value", data=melted_df, palette="Set2", showfliers=True)

    if log_scale:
        ax.set_yscale("log")

    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("")
    plt.ylabel("Value", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    if outputname:
        plt.savefig(outputname, bbox_inches='tight', dpi=100)
    else:
        plt.show()


def plot_description(characteristic : pd.Series, outputname: str=None):
    """
    Plot a description of a characteristic using pandas table.
    """
    plot = plt.subplot(111, frame_on=False)

    plot.xaxis.set_visible(False) 
    plot.yaxis.set_visible(False)
    table(plot, characteristic, loc='upper right')
    if outputname:
        plt.savefig(outputname, bbox_inches='tight', dpi=300)
    else:
        plt.show()

def plot_description_values_table(df : pd.DataFrame, title:str=None, outputname: str=None, figsize: tuple = (1, 1)):
    df_formatted = df.copy()
    df_formatted = df_formatted.applymap(lambda x: f"{x:.5f}" if isinstance(x, float) else x)

    _, ax = plt.subplots(figsize=figsize)
    ax.axis('tight')
    ax.axis('off')
    ax.set_title(title if title else "Dataset Description", fontsize=14, weight='bold')

    table = ax.table(cellText=df_formatted.values,
                    colLabels=df_formatted.columns,
                    cellLoc='center',
                    loc='center')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.25, 1.75)

    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4d004d')
        elif row % 2 == 0:
            cell.set_facecolor('#f1f1f2')
        else:
            cell.set_facecolor('#ffffff')

    plt.tight_layout()
    if outputname:
        plt.savefig(outputname, bbox_inches='tight', dpi=100)
    else:
        plt.show()

def plot_barplot_features(dataset: pd.DataFrame, values: list, outputname: str=None, figsize: tuple = (4, 4)):
    for value in values:
        plt.figure(figsize=figsize)
        dataset[value].value_counts().sort_index().plot.bar(width=0.4, color=['skyblue', 'lightcoral'], edgecolor='black')
        plt.xlabel(value, fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        if outputname:
            plt.savefig(outputname.replace('.png', f'_{value}.png'), bbox_inches='tight', dpi=100)
        else:
            plt.show()

def plot_correlation_matrix(correlation_matrix: pd.DataFrame, outputname:str = None, figsize:tuple=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(correlation_matrix, vmin=-1, vmax=1, cmap='coolwarm')
    fig.colorbar(cax, fraction=0.05, pad=0.04)

    ticks = np.arange(0, len(correlation_matrix.columns), 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(correlation_matrix.columns, rotation=90)
    ax.set_yticks(ticks)
    ax.set_yticklabels(correlation_matrix.columns)
    plt.tight_layout()
    if outputname:
        plt.savefig(outputname, bbox_inches='tight', dpi=100)
    else:
        plt.show()

def plot_chi_pvals_matrix(results_df: pd.DataFrame, outputname:str, figsize: tuple = (10, 8), value: str = None):
    pivot = results_df.pivot(index='Var1', columns='Var2', values=value)
    plt.figure(figsize=figsize)
    sns.heatmap(pivot, annot=True, cmap='coolwarm', center=0.05, cbar_kws={'label': value})
    plt.title("Chi-Squared Test")
    plt.tight_layout()
    if outputname:
        plt.savefig(outputname, bbox_inches='tight', dpi=100)
    else:
        plt.show()

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
