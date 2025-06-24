# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("rf_model.pkl")

features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
            'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean',
            'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se',
            'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se',
            'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst',
            'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
            'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst',
            'fractal_dimension_worst']

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", features=features, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    input_data = [float(request.form[f]) for f in features]
    df = pd.DataFrame([input_data], columns=features)
    pred = model.predict(df)[0]
    prediction = "Malignant (Cancerous)" if pred == "M" else "Benign (Non-Cancerous)"
    return render_template("index.html", features=features, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
