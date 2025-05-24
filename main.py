import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import seaborn as sns
import sys
import numpy as np
from labelsDict import HeartDisease, Pirvision
from heart_dataset_eda import heart_eda_statistics
from pirvision_dataset_eda import pirvision_eda_statistics
from preprocessing_dataset import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight


HEART_DATASET_PATH_TRAIN        = 'datasets/heart_4_train.csv'
HEART_DATASET_PATH_TEST         = 'datasets/heart_4_test.csv'
PIRVISION_DATASET_PATH_TRAIN    = 'datasets/pirvision_office_train.csv'
PIRVISION_DATASET_PATH_TEST     = 'datasets/pirvision_office_test.csv'

def main():
    index = sys.argv[1]
    
    if index == '0':
        # heart_eda_statistics()
        X_train, X_test, y_train, y_test = preprocessing(HeartDisease,
                                                          train_path=HEART_DATASET_PATH_TRAIN, test_path=HEART_DATASET_PATH_TEST,
                                                          imputation='median', quantile1=0.25, quantile3=0.75,
                                                          scaler_type='standard', correlation_factor=0.9)
    elif index == '1':
        # pirvision_eda_statistics()
        X_train, X_test, y_train, y_test = preprocessing(Pirvision, train_path=PIRVISION_DATASET_PATH_TRAIN, test_path=PIRVISION_DATASET_PATH_TEST, imputation='mean')

    print(np.unique(y_train))
    print(np.unique(y_train, return_counts=True))

    classes = np.unique(y_train)

    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train.values.reshape(-1))
    class_weight_dict = dict(zip(classes, class_weights))

    print(class_weight_dict)

    clf = DecisionTreeClassifier(
        max_depth=10,
        min_samples_leaf=5,
        criterion='gini',
        class_weight=class_weight_dict
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))



if __name__=='__main__':
    main()
