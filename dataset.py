"""
Load and preprocess data
"""

import pandas as pd


def fill_age(data: pd.DataFrame):
    """fill missing values in age column

    Args:
        data (pd.DataFrame): data to fill missing values
    """

    # define age values to fill
    age_vals = {
        1: { "male": 32.552, "female": 26.4 },
        2: { "male": 28.402, "female": 24.0 },
        3: { "male": 15.136, "female": 13.2 }
    }

    # fill missing vaules in Age column
    nan_train = data.isna()
    for i in range(len(data)):
        if nan_train.loc[i, 'Age']:
            data.loc[i, 'Age'] = age_vals[data.loc[i, 'Pclass']][data.loc[i, 'Sex']]


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train and test data

    Args:
        train_path (str): path of train data
        test_path (str): path of test data

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: InputData, LabelData, TestData
    """

    # load train and test data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # remove unimportant features in train and test data
    train_df = train_df.drop(["PassengerId, Name, Ticket, Cabin"], axis=1)
    test_df = test_df.drop(["PassengerId, Name, Ticket, Cabin"], axis=1)

    # fill missing values
    # fill age value
    fill_age(train_df)
    fill_age(test_df)
    # fill Embarked
    train_df["Embarked"] = train_df["Embarked"].fillna(
        train_df["Embarked"].value_counts().keys()[0]
    )
    # fill Fare
    test_df["Fare"] = test_df["Fare"].fillna(test_df["Fare"].mean())

    # Convert categorical data to numerical data
    train_df = pd.get_dummies(train_df)
    test_df = pd.get_dummies(test_df)

    # Split train data into input and label data
    input_data = train_df.drop(["Survived"], axis=1)
    output_data = train_df["Survived"]

    return input_data, output_data, test_df
