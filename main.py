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
    discrete_values = [col for col in dataset if dictLabled[col]==labelsDict.DISCRETE]

    # Save description for continue features
    dataset[continue_values].describe().T.reset_index().to_csv(description_csv, index = False)

    # Save description for discrete features

   

if __name__=='__main__':
    main()
