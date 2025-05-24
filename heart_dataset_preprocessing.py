def heart_preprocessing():
    # Load the dataset
    df_train = pd.read_csv(HEART_DATASET_PATH_TRAIN)
    df_test = pd.read_csv(HEART_DATASET_PATH_TEST)


    dataset = pd.concat([df_train, df_test], ignore_index=True)

    dataset.to_csv(os.path.join(os.getcwd(), 'datasets', 'heart_combined.csv'), index=False)

    # Preprocess the dataset
    __preprocess_heart_dataset(dataset.copy())