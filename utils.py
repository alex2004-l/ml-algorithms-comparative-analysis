import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import table

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
        plt.savefig(outputname, bbox_inches='tight', dpi=500)
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
        plt.savefig(outputname, bbox_inches='tight', dpi=300)
    else:
        plt.show()
