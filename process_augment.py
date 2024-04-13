import pandas as pd
from ctgan import CTGAN
from utils import transform_assets

df_train = pd.read_csv("./train.csv")

EPOCHS = 1000
SAMPLES = 2000

print("Consitiuencies in train data are", len((df_train['Constituency ∇'])))
print("Unique Consitiuencies in train data are", len(set(df_train['Constituency ∇'])))

my_df = df_train.copy()
my_df = my_df.drop(['Candidate', 'Constituency ∇', 'ID'], axis = 1)
my_df['Total Assets'] = my_df['Total Assets'].apply(transform_assets)
my_df['Liabilities'] = my_df['Liabilities'].apply(transform_assets)

my_df.to_csv("my_df.csv")

# Names of the columns that are discrete
discrete_columns = my_df.columns

ctgan = CTGAN( epochs = EPOCHS )
ctgan.fit(my_df, discrete_columns)

# Create synthetic data
synthetic_data = ctgan.sample(SAMPLES)
final_df = pd.concat([my_df,synthetic_data])

final_df.to_csv("new_final_df.csv")