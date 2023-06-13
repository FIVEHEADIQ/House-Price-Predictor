import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import messagebox

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

# Create the GUI
root = tk.Tk()
root.title("House Price Prediction")

# Function to predict the house price
def predict_price():
    address = address_entry.get()
    details = details_entry.get()
    
    try:
        address_score = address_encoder.transform([address])[0]
    except ValueError:
        messagebox.showerror('Error', 'Invalid address')
        return
    
    try:
        details_score = details_encoder.transform([details])[0]
    except ValueError:
        messagebox.showerror('Error', 'Invalid details')
        return

    new_data = pd.DataFrame({'address_score': [address_score], 'details_score': [details_score]})

    predicted_value = model.predict(new_data)[0]
    rounded_prediction = round(predicted_value, 2)
    formatted_prediction = 'C${:,}'.format(rounded_prediction)
    messagebox.showinfo('Price Prediction', 'The predicted price of the house is {}'.format(formatted_prediction))

# Address
address_label = tk.Label(root, text="Address:")
address_label.pack()
address_entry = tk.Entry(root)
address_entry.pack()

# Details
details_label = tk.Label(root, text="Details:")
details_label.pack()
details_entry = tk.Entry(root)
details_entry.pack()

# Predict button
predict_button = tk.Button(root, text="Predict Price", command=predict_price)
predict_button.pack()

root.mainloop()
