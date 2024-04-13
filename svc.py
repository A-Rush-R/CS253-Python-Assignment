import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from utils import decode_categorical_data


def svc(train_data_stand, train_labels, test_data_stand) -> pd.DataFrame :

    X_train, X_val, y_train, y_val = train_test_split(train_data_stand, train_labels, test_size = 0.2, random_state = 42)

    # Define the hyperparameters to tune
    params = {
        'C': [ 0.001, 0.1, 1, 10, 100, 1000],  # Regularization parameter
        'loss': ['hinge', 'squared_hinge'],  # Loss function
        'tol': [100, 10,1,1e-1, 1e-2,1e-3],  # Tolerance for stopping optimization
    }

    # Define the model and the scoring metric
    model = LinearSVC(max_iter = 5000)  # Set maximum iterations to avoid potential convergence issues
    scoring = 'f1_weighted'

    # Perform the grid search and tune the hyperparameters
    grid_search = GridSearchCV(model, params, scoring=scoring, cv=5, n_jobs=-1, verbose=0)
    # grid_search.fit(X_train, y_train.idxmax(axis = 1))
    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters and the F1 score on the validation set
    print('Best hyperparameters:', grid_search.best_params_)


    y_train_pred = grid_search.predict(X_train)
    print('F1 score on training set:', f1_score(y_train, y_train_pred, average='weighted', zero_division=1))
    y_val_pred = grid_search.predict(X_val)
    print('F1 score on validation set:', f1_score(y_val, y_val_pred, average='weighted', zero_division=1))
    # print(y_pred)

    y_test_pred = grid_search.predict(test_data_stand)
    y_test_pred = pd.DataFrame({'ID' : range(len(y_test_pred)), 'Education' : decode_categorical_data(y_test_pred)})
    y_test_pred.to_csv("submission_svc.csv", index=False)

    print("Results saved successfully to submission_svc.csv")

    return y_test_pred
