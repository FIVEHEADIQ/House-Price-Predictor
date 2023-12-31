# House Price Prediction

This is a Python module that uses machine learning to predict house prices based on address and details. The module utilizes a gradient boosting regression algorithm to train a model on a dataset of house prices. It takes into account the address and specific details of each house to make accurate price predictions. The model is trained on housing data from various data sets to learn patterns and relationships between the address and details of the house and the price.

## Installation and Usage

To use this module, you'll need Python 3.10.11 or higher installed. You can install the required dependencies by running the following command: pip install -r requirements.txt

1. Run the main file.
2. Input your desired address and details and click the "Predict Price" button in the GUI.
3. See your predicted price in a message box. 
4. Repeat if needed with different addresses and details.

## Important Notes
Please note that the model has an error percentage of around 29%, also note that the prices used for the training data are prices taken from March of 2021, with the current real estate market trends for the GTA (Greater Toronto Area), you can assume the price has gone down by around 11%. NOTE: The data for this model is only for GTA homes, entering any value for addresses or details not in the database will result in an error shown. The data set can be seen in the main.py file. 

## Upcoming Updates
Planned update for July 18th 4953: Can predict price of any valid inputted address and details, as well with inflation predictions and depreciation predictions.
