import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from utils import decode_categorical_data


def knn(train_data_stand, train_labels, test_data_stand) -> pd.DataFrame :

    X_train, X_val, y_train, y_val = train_test_split(train_data_stand, train_labels, test_size = 0.2, random_state = 42)

    # Define the hyperparameters to tune
    params = {
    'n_neighbors': [3, 5, 13, 19, 21, 23, 25, 29],
    'metric': ['euclidean', 'manhattan', 'minkowski'],  # Distance metrics
    'p': [1, 2, 3, 4],  # Minkowski distance parameter (if using 'minkowski' metric)
    'weights': ['uniform', 'distance']  # Weighting scheme for neighbors
    }

    # Define the model and the scoring metric
    model = KNeighborsClassifier()
    scoring = 'f1_weighted'  # You can choose other metrics like F1 score

    # Load your training data (X_train) and target labels (y_train)

    # Perform the grid search and tune the hyperparameters
    grid_search = GridSearchCV(model, params, scoring=scoring, cv=5, n_jobs=-1, verbose=0)

    grid_search.fit(X_train, y_train)

    # Print the best hyperparameters and the accuracy on the validation set
    # Print the best hyperparameters and the F1 score on the validation set
    print('Best hyperparameters:', grid_search.best_params_)
    y_train_pred = grid_search.predict(X_train)
    print('F1 score on training set:', f1_score(y_train, y_train_pred, average='weighted', zero_division=1))
    y_val_pred = grid_search.predict(X_val)
    print('F1 score on validation set:', f1_score(y_val, y_val_pred, average='weighted', zero_division=1))


    y_test_pred = grid_search.predict(test_data_stand)
    y_test_pred = pd.DataFrame({'ID' : range(len(y_test_pred)), 'Education' : decode_categorical_data(y_test_pred)})
    y_test_pred.to_csv("submission_knn.csv", index=False)

    print("Results saved successfully to submission_knn.csv")

    return y_test_pred