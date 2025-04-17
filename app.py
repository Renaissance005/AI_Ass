from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the model and training columns
model = joblib.load("xgboost_model.pkl")
model_columns = joblib.load("model_columns.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract numeric and categorical fields
        G1 = data["G1"]
        G2 = data["G2"]
        absences = data["absences"]
        famrel = data["famrel"]
        schoolsup = data["schoolsup"]
        reason = data["reason"]
        Fjob = data["Fjob"]
        activities = data["activities"]
        romantic = data["romantic"]

        # Calculate derived feature
        G1_G2_avg = (G1 + G2) / 2

        # Construct input DataFrame
        input_df = pd.DataFrame([{
            "G1_G2_avg": G1_G2_avg,
            "G2": G2,
            "absences": absences,
            "famrel": famrel,
            "schoolsup": schoolsup,
            "reason": reason,
            "Fjob": Fjob,
            "activities": activities,
            "romantic": romantic
        }])

        # One-hot encode categorical values
        input_df = pd.get_dummies(input_df)

        # Align with trained model columns
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction = model.predict(input_df)[0]

        # Return result
        return jsonify({"predicted_G3": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
