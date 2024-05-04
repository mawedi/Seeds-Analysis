from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd

# Importing the dataset
def import_dataset(data_path):
    dataset = pd.read_csv(data_path)
    return dataset


# Verifying the dataset
def verify_dataset(df):
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df['Class(1,2,3)'].value_counts())


# Repairing Classes
def repair_class(df):
    # Renaming the column of the targets
    df.rename(columns={'Class(1,2,3)': 'Target'}, inplace=True)

    # Changing the name of the targets
    df['Target'] = df['Target'].replace(1, 'Kama')
    df['Target'] = df['Target'].replace(2, 'Rosa')
    df['Target'] = df['Target'].replace(3, 'Canadian')

    # Verifying the changes
    print(f"Classes: {df['Target'].unique()}")
    print("mohamed", df.columns)

# Separating the dataset to features and target
def split_dataset(df):
    column_to_remote = 'Target'

    # Extracting the columns
    Y = df.pop(column_to_remote)
    X = df

    # Verifying the separation
    print("Features: ", X.head())
    print("Target: ", Y.head())

    return X, Y


# Scaling the features
def scale_features(X): 
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X     


# Splitting the dataset to training and testing
def split_dataset_to_train_and_test(X, Y, test_size=0.25):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)

    return X_train, X_test, Y_train, Y_test

