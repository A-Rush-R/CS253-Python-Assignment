from imblearn.over_sampling import SMOTE
smote = SMOTE()
import pandas as pd
from pre_processing import preprocess_df

def smote_df(dataset_path = "./train.csv") :
    X, y, _ = preprocess_df(dataset_path)
    ## currently using standardized data

    print("old length is", len(y))
    X_smote, y_smote = smote.fit_resample(X.values, y)
    print("new length is", len(y))

    X_smote = pd.DataFrame(X_smote, columns = X.columns)

    return X_smote,y_smote


