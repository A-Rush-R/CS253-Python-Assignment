from dt import dt
from knn import knn
from nbc import nbc
from rf import rf
from svc import svc
from pre_processing import preprocess_df

import argparse

TRAIN_DATASET = "./new_final_df.csv"

def main():
    parser = argparse.ArgumentParser(description="Predict on test data")
    parser.add_argument("model_name", type=str, nargs='?', default="svc",
                        help="Name of the model to use for prediction (default: svc)")
    parser.add_argument("test_file", type=str, nargs='?', default="test.csv",
                        help="Path to the test data file (default: test.csv)")
    args = parser.parse_args()

    # Preprocessing 

    train_data, train_labels, test_data = preprocess_df(args.test_file, TRAIN_DATASET)

    # Load model
    if args.model_name == "dt" : 
        y_pred = dt(train_data, train_labels, test_data)
    elif args.model_name == "knn" :
        y_pred = knn(train_data, train_labels, test_data)
    elif args.model_name == "nbc" :
        y_pred = nbc(train_data, train_labels, test_data)
    elif args.model_name == "rf" :
        y_pred = rf(train_data, train_labels, test_data)
    elif args.model_name == "svc" :
        y_pred = svc(train_data, train_labels, test_data)
    else :
        print("Invalid Model !") 
        return 

if __name__ == "__main__":
    main()
