import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import sys
import labelsDict

HEART_DATASET_PATH         = 'datasets/heart_4_train.csv'
PIRVISION_DATASET_PATH     = 'datasets/pirvision_office_train.csv'
HEART_DESCRIPTION_PATH     = 'description/heart.csv'
PIRVISION_DESCRIPTION_PATH = 'description/pirvision.csv'

def plot_boxplot_value_range(dataset : pd.DataFrame, continue_values : list, outputname: str):
    boxprops = dict(linestyle='-', linewidth=1.5, color='darkblue', facecolor='lightblue')

    plt.figure(figsize=(18, 6))
    dataset.boxplot(
        column=continue_values,
        fontsize=10,
        patch_artist=True,         # Fills boxes with color
        boxprops=boxprops
    )

    plt.xticks(rotation=45)
    plt.title(f"Continuous Variables", fontsize=14)
    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    # plt.savefig(outputname)

def plot_description(characteristic : pd.Series, outputname: str):
    plot = plt.subplot(111, frame_on=False)

    plot.xaxis.set_visible(False) 
    plot.yaxis.set_visible(False)
    table(plot, characteristic, loc='upper right')

    plt.show()
    # plt.savefig(outputname)

def plot_description_values_table(df : pd.DataFrame, outputname: str):
    df_formatted = df.copy()
    df_formatted = df_formatted.applymap(lambda x: f"{x:.5f}" if isinstance(x, float) else x)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.axis('tight')
    ax.axis('off')
    ax.set_title("Heart Disease Dataset Description", fontsize=14, weight='bold')

    # Create the table
    table = ax.table(cellText=df_formatted.values,
                    colLabels=df_formatted.columns,
                    cellLoc='center',
                    loc='center')

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.25, 1.75)

    # Color headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4d004d')
        elif row % 2 == 0:
            cell.set_facecolor('#f1f1f2')  # Light gray row
        else:
            cell.set_facecolor('#ffffff')  # White row

    plt.tight_layout()
    plt.show()
    # plt.savefig(outputname)

def main():
    filename = sys.argv[1]
    dictLabled = {}
    description_csv = ''

    if filename == HEART_DATASET_PATH:
        dictLabled = labelsDict.HeartDisease
        description_csv = HEART_DESCRIPTION_PATH
    elif filename == PIRVISION_DATASET_PATH:
        dictLabled = labelsDict.Pirvision
        description_csv = PIRVISION_DESCRIPTION_PATH
    else:
        print("Unknown set. Please label columns beforehand")
        return -1
    
    dataset = pd.read_csv(filename)

    continue_values = [col for col in dataset if dictLabled[col]==labelsDict.CONTINUE]
    print(f"Continue values: {continue_values}")
    discrete_values = [col for col in dataset if dictLabled[col]==labelsDict.DISCRETE]
    print(f"Discrete values: {discrete_values}")


    # Save description for continue features
    df = dataset[continue_values].describe().T.reset_index()
    df.to_csv(description_csv, index = False)
    # Save description for discrete features
    print(dataset[discrete_values].count().T.reset_index())

    for value in discrete_values:
        print(f"Unique values for {value}: {dataset[value].unique()}")

   

if __name__=='__main__':
    main()
