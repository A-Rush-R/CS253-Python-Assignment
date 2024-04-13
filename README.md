# CS253 Python Assignment

## Task 
You are provided with a training dataset and a test dataset. Your main task is to train a machine-learning model of your choice (SVM, KNN, DecisionTree, RandomForest, or any other) on this dataset and perform multi-class classification. You can only use the libraries mentioned in the Rules section. No deep learning methods/libraries are allowed (For example- ANN, Pytorch, Keras, Tensorflow, etc).

## Contents 

- [`main.py`](./main.py) file containing the driver funtion for the program.
- [`pre_processing.py`](/pre_processing.py) file containing function for pre-processing the dataset. 
- [`process_augment.py`](/process_augment.py) file contaiting function for using ct-gan to genrate synthetic data from train data, for data augmentation.
- [`smote.py`](/smote.py) file containing the function for using smote for oversampling, to tackle data imbalance.
- [`undersample.py`](/undersample.py) file contining the function for undersamling the dataset to achieve equal representation of each target class
- The files for the following models, implemented using grid search &rarr;
    - [`dt.py`](/dt.py) : Decision Tree
    - [`knn.py`](/knn.py) : K-Nearest Neighbours 
    - [`nbc.py`](/nbc.py) : Naive Bayes Classifier
    - [`rf.py`](/rf.py) : Random Forest 
    - [`svc.py`](/svc.py) : Support Vector Classifier
- [`EDA.ipynb`](/EDA.ipynb) file containing the notebook for data analysis and plots.
- [`utils.py`](/utils.py) file containing the utility functions.
- [`my_df`](/my_df.csv) file contating the preprocessed train dataset.
- [`new_final_df`](/new_final_df.csv) file containing the original data with the augmented data, using CTGAN.
- [`final_submission_1.csv`](/final_submission_1.csv) containing the second best of all submissions, generated using SVC model, having a public score of **0.23874** and a private score of 0.24618.
- [`final_submission_2.csv`](/final_submission_2.csv) containing the best of all submissions, generated using SVC model, having a public score of **0.22828** and a private score of  **0.25139**.
- [`dt.csv`](/dt.csv) containg the third submission, trained using Decision Tree model.

## Instructions

- First of all install all the required libraries using the following command 

```shell
pip install -r requierments.txt
```
- (Optional) Run the following command to process and augment the train dataset 
```shell
python process_augment.csv
```
- For convinience, (and because of time-taking training of CTGAN), augmented data is already generated and provided as `new_final_df.csv` 
- Next make sure the data you want to test is in the root directory and run the following command 

```shell
python main.py <model_name> <test_file.csv>  
```
- If no `test_file` is mentioned then it is set to `test.csv` by default 
- If no `model_name` is mentioned then SVC is used by default 

- Model name has to be on among those mentioned in the contents section (mention without `.py`)