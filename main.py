import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import table
import seaborn as sns
import sys
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from labelsDict import HeartDisease, Pirvision
from heart_dataset_eda import heart_eda_statistics
from pirvision_dataset_eda import pirvision_eda_statistics
from preprocessing_dataset import preprocessing

HEART_DATASET_PATH_TRAIN        = 'datasets/heart_4_train.csv'
HEART_DATASET_PATH_TEST         = 'datasets/heart_4_test.csv'
PIRVISION_DATASET_PATH_TRAIN    = 'datasets/pirvision_office_train.csv'
PIRVISION_DATASET_PATH_TEST     = 'datasets/pirvision_office_test.csv'

def plot_confusion_matrix(cm, outputname = None):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Reds')
    if outputname:
        plt.savefig(outputname, dpi = 100)
    plt.show()

def get_report(name, report):
    
    row_data = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

    return pd.DataFrame([row_data], index=[name])


def decision_tree_classifier(X_train, X_test, y_train, y_test):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
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

    print("Accuracy:", accuracy_score(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

    # confusion = confusion_matrix(y_test, y_pred)
    # plot_confusion_matrix(confusion, "confusion/decision_tree_pirvision.png")
    report = classification_report(y_test, y_pred, output_dict=True)

    return get_report("decision_tree", report)


def random_forest_classifier(X_train, X_test, y_train, y_test):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    

    rf_model = RandomForestClassifier(
        n_estimators=200,          
        max_depth=12,                 
        min_samples_leaf=5,
        criterion='gini',
        class_weight=class_weight_dict,
        max_samples=0.7,
        max_features=0.6,
        random_state=42
    )

    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

    # confusion = confusion_matrix(y_test, y_pred)
    # plot_confusion_matrix(confusion, "confusion/random_forest_pirvision.png")
    report = classification_report(y_test, y_pred, output_dict=True)

    return get_report("random_forest", report)



def mlp_classifier(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=5000,
        batch_size=128,
        alpha=0.001,
        random_state=42
    )

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

    # confusion = confusion_matrix(y_test, y_pred)
    # plot_confusion_matrix(confusion, "confusion/mlp_pirvision.png")
    report = classification_report(y_test, y_pred, output_dict=True)

    return get_report("mlp", report)



def nll(Y, T, w, lam=0.01):
    N = T.shape[0]
    return -1/N * np.sum(T * np.log(Y) + (1 - T) * np.log(1 - Y)) + lam * np.sum(w**2)

def logistic_regression_classifier(X1, X2, y_train, y_test):
    X_train = np.hstack([np.ones((X1.shape[0], 1)), X1])
    X_test = np.hstack([np.ones((X2.shape[0], 1)), X2])

    unique_classes = np.unique(y_train)

    prob_matrix = np.zeros((X_test.shape[0], len(unique_classes)))
    
    if len(unique_classes) > 2:
        for y_class in unique_classes:
            y_encoded_train_modified = (y_train==y_class).astype(int)
            w = train_and_eval_logistic(X_train, y_encoded_train_modified, lr = 0.01, epochs_no=1000)
            y_pred_probs = predict_logistic(X_test, w)
            prob_matrix[:, y_class] = y_pred_probs

        y_pred = np.argmax(prob_matrix, axis=1)
    else:
        w = train_and_eval_logistic(X_train, y_train, lr = 0.01, epochs_no=1000)
        y_pred_probs = predict_logistic(X_test, w)
        y_pred = (y_pred_probs >= 0.5).astype(int)

    print("Accuracy:",accuracy_score(y_pred, y_test))
    # print(classification_report(y_test, y_pred))

    report = classification_report(y_test, y_pred, output_dict=True)

    return get_report("logistic_regression", report)


def logistic(x):
    return 1 / (1 + np.exp(-x))

def predict_logistic(X, w):
    return logistic(X @ w)

def train_and_eval_logistic(X_train, T_train,
                            lr=0.01, epochs_no=100):

    N_train, D = X_train.shape
    w = np.random.randn(D)

    for epoch in range(epochs_no):
        Y_train = predict_logistic(X_train, w)

        grad = (1 / N_train) * X_train.T @ (Y_train - T_train)
        w = w - lr * grad
    return w


def main():
    index = sys.argv[1]
    ml_type = sys.argv[2]
    
    if index == '0':
        # heart_eda_statistics()
        X_train, X_test, y_train, y_test = preprocessing(HeartDisease,
                                                          train_path=HEART_DATASET_PATH_TRAIN, test_path=HEART_DATASET_PATH_TEST,
                                                          imputation='most_frequent', quantile1=0.25, quantile3=0.75,
                                                          scaler_type='standard', correlation_factor=0.9)
    elif index == '1':
        # pirvision_eda_statistics()
        X_train, X_test, y_train, y_test = preprocessing(Pirvision,
                                                         train_path=PIRVISION_DATASET_PATH_TRAIN, test_path=PIRVISION_DATASET_PATH_TEST,
                                                         imputation='median', quantile1=0.1, quantile3=0.9,
                                                          scaler_type='standard', correlation_factor=0.9)


    label_encoder = LabelEncoder()

    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()

    y_train = label_encoder.fit_transform(y_train)
    y_test  = label_encoder.transform(y_test)

    # if ml_type =='0':
    #     decision_tree_classifier(X_train, X_test, y_train, y_test)
    # elif ml_type == '1':
    #     random_forest_classifier(X_train, X_test, y_train, y_test)
    # elif ml_type == '3':
    #     mlp_classifier(X_train, X_test, y_train, y_test)
    # elif ml_type == '2':
    #     logistic_regression_classifier(X_train, X_test, y_train, y_test)
    # else:
    #     raise ValueError("Unknown second argument")

    df1 = decision_tree_classifier(X_train, X_test, y_train, y_test)
    df2 = random_forest_classifier(X_train, X_test, y_train, y_test)
    df3 = logistic_regression_classifier(X_train, X_test, y_train, y_test)
    df4 = mlp_classifier(X_train, X_test, y_train, y_test)

    results_df = pd.concat([df1, df2, df3, df4])

    latex_table = results_df.round(3).to_latex()
    print(latex_table)


if __name__=='__main__':
    main()
