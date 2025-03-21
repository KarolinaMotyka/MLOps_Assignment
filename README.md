# MLOps_Assignment: Penguin Species Prediction

##  Overview
This project is an **automated MLOps pipeline** that fetches new penguin data daily, predicts its species using a trained machine learning model, and updates the results everyday at 7.30.
---

##  Repository Structure

## `.github/workflows/`
- **`predict.yml`** → GitHub Action workflow that runs **daily at 7:30 AM UTC** to fetch new data, make a prediction, and push the result.

###  data/
- **`penguins.db`** → SQLite database containing the original dataset of penguin species.

### `models/`
- **`penguin_logreg_model.pkl`** → Trained logistic regression model used for predictions.
- **`label_encoder.pkl`** → Encoder that converts species names (`Adelie`, `Chinstrap`, `Gentoo`) into numerical labels.

###  `prediction_output/`
- **`prediction.html`**  → Webpage that displays the latest predicted species, updated daily.


###  Root files
- **`predict.py`** → Main script that:
  - Fetches new penguin data from an API
  - Preprocesses the data
  - Loads the trained model
  - Predicts the species
  - Saves the result as an HTML file 
- **`train_model.ipynb`** → notebook used to train and save the model.

---

##  How It Works
1. **Training:** The model is trained in `train_model.ipynb` using historical penguin data.
2. **Daily Predictions:**
   - GitHub Actions runs `predict.py` every morning.
   - The script fetches **new penguin data** from the API.
   - The model predicts the species.
   - The result is saved in `prediction_output/prediction.html`.
   - GitHub Actions commits and pushes the updated file to GitHub Pages.
3. **Viewing Predictions:** The prediction is displayed on **GitHub Pages**.

---

