# model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# Load the dataset
csv_files = ['https://raw.githubusercontent.com/kennymkchan/greater-toronto-area-housing-data/master/data/oakville/csv/oakville_housing_data_03222021.json.csv',
            'https://raw.githubusercontent.com/kennymkchan/greater-toronto-area-housing-data/master/data/mississauga/csv/mississauga_housing_data_03222021.json.csv',
            'https://raw.githubusercontent.com/kennymkchan/greater-toronto-area-housing-data/master/data/toronto/csv/toronto_housing_data_03222021.json.csv',
            'https://raw.githubusercontent.com/kennymkchan/greater-toronto-area-housing-data/master/data/ajax/csv/ajax_housing_data_03222021.json.csv',
            'https://raw.githubusercontent.com/kennymkchan/greater-toronto-area-housing-data/master/data/aurora/csv/aurora_housing_data_03222021.json.csv',
            'https://raw.githubusercontent.com/kennymkchan/greater-toronto-area-housing-data/master/data/brampton/csv/brampton_housing_data_03222021.json.csv',
            'https://raw.githubusercontent.com/kennymkchan/greater-toronto-area-housing-data/master/data/burlington/csv/burlington_housing_data_03222021.json.csv',
            'https://raw.githubusercontent.com/kennymkchan/greater-toronto-area-housing-data/master/data/markham/csv/markham_housing_data_03222021.json.csv',
            'https://raw.githubusercontent.com/kennymkchan/greater-toronto-area-housing-data/master/data/new_market/csv/new_market_housing_data_03222021.json.csv',
            'https://raw.githubusercontent.com/kennymkchan/greater-toronto-area-housing-data/master/data/oshawa/csv/oshawa_housing_data_03222021.json.csv',
            'https://raw.githubusercontent.com/kennymkchan/greater-toronto-area-housing-data/master/data/pickering/csv/pickering_housing_data_03222021.json.csv',
            'https://raw.githubusercontent.com/kennymkchan/greater-toronto-area-housing-data/master/data/richmond_hill/csv/richmond_hill_housing_data_03222021.json.csv',
            'https://raw.githubusercontent.com/kennymkchan/greater-toronto-area-housing-data/master/data/vaughan/csv/vaughan_housing_data_03222021.json.csv',
            'https://raw.githubusercontent.com/kennymkchan/greater-toronto-area-housing-data/master/data/whitby/csv/whitby_housing_data_03222021.json.csv']
dataframes = []

for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)
combined_df = pd.concat(dataframes)

# Convert all price values into usable float values
def remove_non_numeric(value):
    final_value = ''.join(char for char in str(value) if char.isdigit())
    if final_value == '':
        return 0
    return float(final_value)

combined_df['price'] = combined_df['price'].apply(remove_non_numeric)
combined_df['price'] = pd.to_numeric(combined_df['price'])

# Initialize scoring function
address_encoder = LabelEncoder()
details_encoder = LabelEncoder()

combined_df['address_score'] = address_encoder.fit_transform(combined_df['address'])
combined_df['details_score'] = details_encoder.fit_transform(combined_df['details'])

X = combined_df[['address_score', 'details_score']]
Y = combined_df['price']

model = GradientBoostingRegressor()
model.fit(X, Y)

def predict_price(address: str, details: str) -> float:
    """
    Predicts the price of a house based on the given address and details.

    Args:
        address (str): The address of the house.
        details (str): Additional details of the house.

    Returns:
        float: The predicted price of the house.
    """
    try:
        address_score = address_encoder.transform([address])[0]
    except ValueError:
        raise ValueError("Invalid address")

    try:
        details_score = details_encoder.transform([details])[0]
    except ValueError:
        raise ValueError("Invalid details")

    new_data = pd.DataFrame({'address_score': [address_score], 'details_score': [details_score]})
    predicted_value = model.predict(new_data)[0]

    return round(predicted_value, 2)

# Calculate mean absolute error
predicted_prices = model.predict(X)
mae = mean_absolute_error(Y, predicted_prices)
mae_percent = (mae / np.mean(Y)) * 100
print("Mean Absolute Error (%):", mae_percent)
