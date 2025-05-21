import matplotlib.pyplot as plt
import pandas as pd
import sys
import labelsDict

HEART_DATASET_PATH     = 'datasets/heart_4_train.csv'
PIRVISION_DATASET_PATH = 'datasets/pirvision_office_train.csv'

def plot_boxplot(dataset : pd.DataFrame, continue_values : list):
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


def main():
    filename = sys.argv[1]
    dictLabled = {}

    if filename == HEART_DATASET_PATH:
        dictLabled = labelsDict.HeartDisease
    elif filename == PIRVISION_DATASET_PATH:
        dictLabled = labelsDict.Pirvision
    else:
        print("Unknown set. Please label columns beforehand")
        return -1
    
    dataset = pd.read_csv(filename)

    # if filename == PIRVISION_DATASET_PATH:
    #     dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'], errors='coerce')
    #     dataset['Day'] = pd.to_datetime(dataset['Day'], errors='coerce')
    #     print(dataset['Day'])
    #     print(dataset['Timestamp'])

    continue_values = [col for col in dataset if dictLabled[col]==labelsDict.CONTINUE]
    discrete_values = [col for col in dataset if dictLabled[col]==labelsDict.DISCRETE]

    for value in continue_values:
        print(dataset[value].describe())
    
   

if __name__=='__main__':
    main()
