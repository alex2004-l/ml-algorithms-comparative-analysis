import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import seaborn as sns
import sys
from heart_dataset_eda import heart_eda_statistics
from pirvision_dataset_eda import pirvision_eda_statistics

def main():
    index = sys.argv[1]
    
    if index == '0':
        heart_eda_statistics()
    elif index == '1':
        pirvision_eda_statistics()

if __name__=='__main__':
    main()
