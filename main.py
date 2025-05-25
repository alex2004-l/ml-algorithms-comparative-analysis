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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


HEART_DATASET_PATH_TRAIN        = 'datasets/heart_4_train.csv'
HEART_DATASET_PATH_TEST         = 'datasets/heart_4_test.csv'
PIRVISION_DATASET_PATH_TRAIN    = 'datasets/pirvision_office_train.csv'
PIRVISION_DATASET_PATH_TEST     = 'datasets/pirvision_office_test.csv'

def decision_tree_classifier(X_train, X_test, y_train):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train.values.reshape(-1))
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    clf = DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=5,
        criterion='gini',
        class_weight=class_weight_dict,
        random_state=42
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return y_pred

def random_forest_classifier(X_train, X_test, y_train):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train.values.reshape(-1))
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    rf_model = RandomForestClassifier(
        n_estimators=155,          
        max_depth=15,                 
        min_samples_leaf=4,
        criterion='gini',
        class_weight=class_weight_dict,
        max_samples=0.7,
        max_features=0.6,
        random_state=42
    )

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    return y_pred

def mlp_classifier(X_train, X_test, y_train):
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=400,
        batch_size=64,
        alpha=0.0001,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    return y_pred


def main():
    index = sys.argv[1]
    
    if index == '0':
        # heart_eda_statistics()
        X_train, X_test, y_train, y_test = preprocessing(HeartDisease,
                                                          train_path=HEART_DATASET_PATH_TRAIN, test_path=HEART_DATASET_PATH_TEST,
                                                          imputation='most_frequent', quantile1=0.25, quantile3=0.75,
                                                          scaler_type='robust', correlation_factor=0.9)
    elif index == '1':
        # pirvision_eda_statistics()
        X_train, X_test, y_train, y_test = preprocessing(Pirvision,
                                                         train_path=PIRVISION_DATASET_PATH_TRAIN, test_path=PIRVISION_DATASET_PATH_TEST,
                                                         imputation='mean', quantile1=0.1, quantile3=0.9,
                                                          scaler_type='standard', correlation_factor=0.9)

    y_pred = decision_tree_classifier(X_train, X_test, y_train)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    y_pred = random_forest_classifier(X_train, X_test, y_train)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    y_pred = mlp_classifier(X_train, X_test, y_train)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))



if __name__=='__main__':
    main()
