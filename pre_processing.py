import pandas as pd
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler ## for normalization
from utils import encode_categorical_data, transform_assets

def encode_column( df : pd.DataFrame , column : str ) -> pd.DataFrame :
    # perform one-hot encoding on the 'color' column
    one_hot = pd.get_dummies(df[column])
    one_hot = one_hot.astype(int)
    # print(one_hot)

    # concatenate the one-hot encoding with the original dataframe
    df1 = pd.concat([df, one_hot], axis=1)

    # drop the original 'color' column
    df1 = df1.drop(column, axis=1)
    
    return df1

def preprocess_df(dataset_path : str, train_path : str = "./my_df.csv") -> pd.DataFrame :

    categorical_columns = ['state','Party']

    encoded_df = pd.read_csv(train_path)

    for column in  categorical_columns :
        encoded_df = encode_column(encoded_df, column)
        
    train_data = encoded_df.drop('Education', axis=1)

    df_test = pd.read_csv(dataset_path)
    test_data = df_test.drop(['Candidate','Constituency ∇','ID'], axis = 1)
    test_data['Total Assets'] = test_data['Total Assets'].apply(transform_assets)
    test_data['Liabilities'] = test_data['Liabilities'].apply(transform_assets)

    for column in  categorical_columns :
        test_data = encode_column(test_data, column)
        
    train_labels = encoded_df['Education']
    train_labels = encode_categorical_data(train_labels)

    ## Normalization 

    # # fit scaler on training data
    # norm = MinMaxScaler().fit(train_data)

    # # transform training data
    # train_data_norm = norm.transform(train_data)

    # # transform testing dataabs
    # test_data_norm = norm.transform(test_data)

    # train_data_norm = pd.DataFrame(train_data_norm, columns=train_data.columns)
    # test_data_norm = pd.DataFrame(test_data_norm, columns=test_data.columns)

    ## Standardization

    # copy of datasets
    train_data_stand = train_data.copy()
    test_data_stand = test_data.copy()

    # numerical features
    num_cols = ['Criminal Case','Total Assets','Liabilities']

    # apply standardization on numerical features
    for i in num_cols:
        
        # fit on training data column
        scale = StandardScaler().fit(train_data_stand[[i]])
        
        # transform the training data column
        train_data_stand[i] = scale.transform(train_data_stand[[i]])
        
        # transform the testing data column
        test_data_stand[i] = scale.transform(test_data_stand[[i]])
        
    train_data_stand = train_data_stand.astype(float)
    test_data_stand = test_data_stand.astype(float)

    return train_data_stand, train_labels, test_data_stand