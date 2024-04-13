from sklearn.metrics import f1_score
import re

education_levels = ['10th Pass', '12th Pass', '5th Pass', '8th Pass', 'Doctorate',
                    'Graduate', 'Graduate Professional', 'Literate', 'Others', 'Post Graduate']
education_order = {'Literate': 0, '5th Pass': 1, '8th Pass': 2, '10th Pass': 3,
                  '12th Pass': 4, 'Graduate': 5, 'Post Graduate': 6,
                  'Graduate Professional': 7, 'Doctorate': 8, 'Others': 9}

def encode_categorical_data(data : list) -> list :
  """
  Encodes categorical data using a provided mapping dictionary.

  Args:
      data (list): A list of categorical values.

  Returns:
      list: A list of encoded numerical values.
  """
  encoded_data = []
  for value in data:
    if value in education_order:
      encoded_data.append(education_order[value])
    else:
      raise ValueError(f"Value '{value}' not found in the mapping dictionary.")
  return encoded_data


def decode_categorical_data(data : list) -> list:
  """
  Decodes numerical data back to categorical values using a mapping dictionary.

  Args:
      data (list): A list of encoded numerical values.

  Returns:
      list: A list of decoded categorical values.
  """
  mapping = {v: k for k, v in education_order.items()} 
  decoded_data = []

  for code in data:
    if code in mapping.keys():  
      decoded_data.append([value for key, value in mapping.items() if key == code][0])
    else:
      raise ValueError(f"Code '{code}' not found in the mapping dictionary.")
  return decoded_data


def evaluate_output(y_train,y_train_pred, y_val, y_val_pred) -> dict :
    train_f1_weighted = f1_score(y_train, y_train_pred, average = 'weighted')
    val_f1_weighted = f1_score(y_val, y_val_pred, average = 'weighted')

    # Append the scores to the dataframe
    return { 'train_f1_weighted': train_f1_weighted, 'val_f1_weighted': val_f1_weighted}

def transform_assets(assets : str, num_word_dict : dict = {
    "Crore" : 10000000,
    "Lac" : 100000,
    "Thou" : 1000,
    "Hund" : 100
}) -> int :
    if assets == "0" :
        return 0
    else :
        match = re.search(r'(\d+)\s*(\w+)', assets)
        num_word = match.group(2)
        num = int(match.group(1))
        return num_word_dict[num_word] * num