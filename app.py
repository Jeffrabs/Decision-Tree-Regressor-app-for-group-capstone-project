
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


st.title("Birth Weight Prediction App")
st.write("Investigating gravidity and birth weight")

# Load dataset
df = pd.read_csv("processed_maternity_data.csv")

# Prepare data
df = df[["Gravidae", "Birth_weight"]].copy()
df["Gravidae"] = pd.to_numeric(df["Gravidae"], errors="coerce")
df["Birth_weight"] = pd.to_numeric(df["Birth_weight"], errors="coerce")
df = df.dropna()

# Define input and output
X = df[["Gravidae"]]
y = df["Birth_weight"]

# Train Decision Tree model
model = DecisionTreeRegressor(
    max_depth=4,
    random_state=42
)
model.fit(X, y)

# User input
gravidae = st.number_input(
    "Enter number of pregnancies (gravidity)",
    min_value=1,
    max_value=20,
    value=1,
    step=1
)

# Make prediction
if st.button("Predict Birth weight"):
    new_data = pd.DataFrame({
        "Gravidae": [gravidae]
    })

    prediction = model.predict(new_data)[0]

    st.success(
        f"Predicted birth weight: {prediction:.0f} grams"
    )
