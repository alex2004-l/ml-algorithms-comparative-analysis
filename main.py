import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import seaborn as sns
import sys
from heart_dataset_eda import heart_eda_statistics
from pirvision_dataset_eda import pirvision_eda_statistics
from heart_dataset_preprocessing import heart_preprocessing
# from pirvision_dataset_preprocessing import pirvision_preprocessing

def main():
    index = sys.argv[1]
    
    if index == '0':
        heart_eda_statistics()
        # heart_preprocessing()
    elif index == '1':
        pirvision_eda_statistics()
        # pirvision_preprocessing()

if __name__=='__main__':
    main()
