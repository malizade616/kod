# fråga 15 kapital 3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
    
# 1. csv kod
df = pd.read_csv("/Users/mehdializadeh/Downloads/Data-science 1/filer/statisk/car_price_dataset.csv", sep=";")
print(df.head())
df = df.dropna()

df_original = df.copy()  
df = pd.get_dummies(df, drop_first=True)

# 5. Definiera X och y 
X = df.drop(columns=["Price"])
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)




print(f"Modellens resultat")
print(f"Felmarginal (RMSE): {rmse:.2f} kr")







#streamlit del
st.title("🚗 Prediktera bilpris")

st.write("Fyll i information om bilen:")

# Inputs
year = st.number_input("Årsmodell", min_value=1990, max_value=2025, value=2015)
mileage = st.number_input("Körsträcka", min_value=0, value=50000)

brand = st.selectbox("Märke", df_original["Brand"].unique())
fuel = st.selectbox("Bränsle", df_original["Fuel_Type"].unique())

# Skapa input-data
input_data = pd.DataFrame({
    "Year": [year],
    "Mileage": [mileage],
    "Brand": [brand],
    "Fuel_Type": [fuel]
})

# One-hot encoding
input_data = pd.get_dummies(input_data)

# Matcha kolumner med träningsdata
input_data = input_data.reindex(columns=X.columns, fill_value=0)

# Prediction
if st.button("Prediktera pris"):
    prediction = model.predict(input_data)
    st.success(f"💰 Predikterat pris: {prediction[0]:,.0f} kr")
