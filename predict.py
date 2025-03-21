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

html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Penguin Prediction</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            text-align: center;
            padding-top: 50px;
            background-color: #f0f8ff;
        }}
        .card {{
            background-color: white;
            padding: 20px;
            margin: auto;
            width: 300px;
            border-radius: 15px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
        }}
    </style>
</head>
<body>
    <div class="card">
        <h1>🐧 Predicted Species:</h1>
        <h2>{species[0]}</h2>
    </div>
</body>
</html>
"""

with open("prediction_output/prediction.html", "w") as f:
    f.write(html)
