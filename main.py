import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import seaborn as sns
import sys
from heart_dataset_eda import solve_first_eda_heart

def main():
    index = sys.argv[1]
    
    if index == '0':
        solve_first_eda_heart()
    elif index == '1':
        pass


if __name__=='__main__':
    main()
