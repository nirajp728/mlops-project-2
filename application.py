from flask import Flask, render_template, request
import joblib
import numpy as np
import os
from sklearn.dummy import DummyClassifier

app = Flask(__name__)

MODEL_PATH = "artifacts/models/model.pkl"

# Ensure directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Load model or create dummy if missing
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print("Warning: model.pkl not found. Using a dummy classifier.")
    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit([[0, 0, 0, 0]], [0])  # Dummy training
    joblib.dump(dummy_model, MODEL_PATH)
    model = dummy_model

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        try:
            sepal_length = float(request.form['SepalLengthCm'])
            sepal_width = float(request.form['SepalWidthCm'])
            petal_length = float(request.form['PetalLengthCm'])
            petal_width = float(request.form['PetalWidthCm'])

            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction = model.predict(input_data)[0]
        except Exception as e:
            prediction = f"Error during prediction: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
