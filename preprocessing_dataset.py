import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
from labelsDict import CONTINUE, DISCRETE, LABEL
from sklearn.experimental import enable_iterative_imputer 
from sklearn.impute import IterativeImputer


def preprocessing(dictionary:dict, train_path:str, test_path:str, imputation:str='mean', quantile1:float=0.25, quantile3:float=0.75, scaler_type:str='standard', correlation_factor:float=0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    # Replacing outliers with Nan
    for col in df_train:
        if dictionary[col] != CONTINUE:
            continue
        Q1 = df_train[col].quantile(quantile1)
        Q3 = df_train[col].quantile(quantile3)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_train[col] = df_train[col].mask((df_train[col] < lower_bound) | (df_train[col] > upper_bound))
        df_test[col] = df_test[col].mask((df_test[col] < lower_bound) | (df_test[col] > upper_bound))

    # Imputation of missing values
    if imputation == 'mean':
        imputer = SimpleImputer(strategy='mean')
    elif imputation == 'median':
        imputer = SimpleImputer(strategy='median')
    elif imputation == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
    elif imputation == 'iterative':
        imputer = IterativeImputer()
    else:
        raise ValueError(f"Unknown imputation method: {imputation}")

    columns = [cols for cols in df_train if dictionary[cols] in [DISCRETE, CONTINUE, LABEL]] 

    df_train[columns] = imputer.fit_transform(df_train[columns])
    df_test[columns]  = imputer.transform(df_test[columns])

    columns = [cols for cols in df_train if dictionary[cols] in [DISCRETE, CONTINUE]]

    corr_matrix = df_train[columns].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_factor)]

    df_train = df_train.drop(columns=to_drop)
    df_test = df_test.drop(columns=to_drop)

    numeric_cols = [cols for cols in df_train.columns if dictionary[cols] in [DISCRETE, CONTINUE]]

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    df_train[numeric_cols] = scaler.fit_transform(df_train[numeric_cols])
    df_test[numeric_cols]  = scaler.transform(df_test[numeric_cols])

    columns = [cols for cols in df_train.columns if dictionary[cols] in [DISCRETE, CONTINUE]]
    labels  = [cols for cols in df_train.columns if dictionary[cols] in LABEL]

    return df_train[columns], df_test[columns], df_train[labels], df_test[labels]

