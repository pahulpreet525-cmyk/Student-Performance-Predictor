from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("models/student_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        hours = float(request.form["hours"])
        attendance = float(request.form["attendance"])
        previous = float(request.form["previous"])

        data = pd.DataFrame({
            "Hours_Studied": [hours],
            "Attendance": [attendance],
            "Previous_Scores": [previous]
        })

        prediction = model.predict(data)[0]
        prediction = max(0, min(100, round(prediction, 2)))  # clamp 0-100

        return render_template("index.html", prediction=prediction)

    except Exception as e:
        return render_template("index.html", error="Something went wrong. Please check your inputs.")

if __name__ == "__main__":
    app.run(debug=True)