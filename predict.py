import pandas as pd
import joblib
import requests
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Fetch new penguin data from API
response = requests.get("http://130.225.39.127:8000/new_penguin/")
data = response.json()
new_data = pd.DataFrame([data])

# Feature Selection
expected_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
new_data = new_data[expected_features]

# Load trained model and label encoder
model = joblib.load("models/penguin_logreg_model.pkl")
labelencoder = joblib.load("models/label_encoder.pkl")

# Predict species
prediction = model.predict(new_data)

# Decode prediction
species = labelencoder.inverse_transform(prediction)


# Save prediction to file
with open("prediction_output/prediction.txt", "w") as f:
    f.write(f"Predicted penguin: {species[0]}")
