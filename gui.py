# gui.py

import tkinter as tk
from tkinter import messagebox
from model import predict_price

def predict_price_gui():
    """
    Retrieves user input from the GUI, calls the prediction function, and displays the predicted price in a message box.

    This function retrieves the user-provided address and details from the GUI text entry fields. It then calls the `predict_price`
    function from the model module to predict the price of the house based on the provided information. The predicted price is
    displayed in a message box.

    If an error occurs during the prediction process, such as an invalid address or details, an error message box is displayed
    to notify the user.

    Returns:
        None
    """

    address = address_entry.get()
    details = details_entry.get()

    try:
        predicted_price = predict_price(address, details)
        messagebox.showinfo('Price Prediction', f'The predicted price of the house is C${predicted_price:,}')
    except ValueError as e:
        messagebox.showerror('Error', str(e))

root = tk.Tk()
root.title("House Price Prediction")

address_label = tk.Label(root, text="Address:")
address_label.pack()
address_entry = tk.Entry(root)
address_entry.pack()

details_label = tk.Label(root, text="Details:")
details_label.pack()
details_entry = tk.Entry(root)
details_entry.pack()

predict_button = tk.Button(root, text="Predict Price", command=predict_price_gui)
predict_button.pack()

root.mainloop()
