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
from sklearn.linear_model import LogisticRegression


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

def logistic(x):
    return 1 / (1 + np.exp(-x))

def nll(Y, T):
    N = T.shape[0]
    return -1/N * np.sum(T * np.log(Y) + (1-T) * np.log(1 - Y))

def predict_logistic(X, w):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    return logistic(X @ w)

def accuracy(Y, T):
    Y = np.array(Y).flatten()
    T = np.array(T).flatten()
    return np.mean((Y >= 0.5) == T)

def train_and_eval_logistic(X_train: pd.DataFrame, T_train: pd.Series,
                            X_test: pd.DataFrame, T_test: pd.Series,
                            lr=0.01, epochs_no=100):
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    T_train = T_train.to_numpy().flatten()
    T_test = T_test.to_numpy().flatten()

    N_train, D = X_train.shape
    w = np.random.randn(D)

    train_acc, test_acc = [], []
    train_nll, test_nll = [], []

    for epoch in range(epochs_no):
        Y_train = predict_logistic(X_train, w)
        Y_test = predict_logistic(X_test, w)

        train_acc.append(accuracy(Y_train, T_train))
        test_acc.append(accuracy(Y_test, T_test))
        train_nll.append(nll(Y_train, T_train))
        test_nll.append(nll(Y_test, T_test))

        grad = (1 / N_train) * X_train.T @ (Y_train - T_train)
        w = w - lr * grad

    return w, train_nll, test_nll, train_acc, test_acc


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

    w, train_nll, test_nll, train_acc, test_acc = train_and_eval_logistic(X_train, y_train, X_test, y_test, lr=0.1, epochs_no=10000)

    y_pred_probs = predict_logistic(X_test, w)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    print("Predicted labels:", np.unique(y_pred, return_counts=True))

    print("Accuracy:",accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))



if __name__=='__main__':
    main()
