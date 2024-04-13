import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from utils import decode_categorical_data


def nbc(train_data_stand, train_labels, test_data_stand) -> pd.DataFrame :

    X_train, X_val, y_train, y_val = train_test_split(train_data_stand, train_labels, test_size = 0.2, random_state = 42)


    pipeline = Pipeline([('clf', GaussianNB())])

    # Define the parameter grid for GridSearchCV
    param_grid = {
        'clf__var_smoothing': [1, 2, 8, 10, 12, 20],  # Smoothing parameter
        'clf__priors': [None]        # Prior probabilities for classes
    }

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)

    # Perform the grid search and tune the hyperparameters
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters and the accuracy on the validation set
    print('Best hyperparameters:', grid_search.best_params_)
    y_train_pred = grid_search.predict(X_train)
    print('F1 score on training set:', f1_score(y_train, y_train_pred, average='weighted', zero_division=1))
    y_val_pred = grid_search.predict(X_val)
    print('F1 score on validation set:', f1_score(y_val, y_val_pred, average='weighted', zero_division=1))


    y_test_pred = grid_search.predict(test_data_stand)
    y_test_pred = pd.DataFrame({'ID' : range(len(y_test_pred)), 'Education' : decode_categorical_data(y_test_pred)})
    y_test_pred.to_csv("submission_nbc.csv", index=False)

    print("Results saved successfully to submission_nbc.csv")

    return y_test_pred