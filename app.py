from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json

app = Flask(__name__)

# ── Train model on startup ──────────────────────────────────────────────────
df = pd.read_csv("Gaming_Academic_Performance.csv")

le_gender = LabelEncoder()
le_genre  = LabelEncoder()
le_stress = LabelEncoder()

df["gender_enc"] = le_gender.fit_transform(df["gender"])
df["genre_enc"]  = le_genre.fit_transform(df["gaming_genre"])
df["stress_enc"] = le_stress.fit_transform(df["stress_level"])

FEATURES = [
    "gaming_hours", "study_hours", "sleep_hours", "attendance",
    "social_activity", "device_usage", "reaction_time_ms",
    "addiction_score", "gender_enc", "genre_enc", "stress_enc", "age"
]

X = df[FEATURES]
y = df["grades"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred   = model.predict(X_test)
R2       = round(r2_score(y_test, y_pred), 4)
RMSE     = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2)
MAE      = round(float(mean_absolute_error(y_test, y_pred)), 2)

# Dashboard data
importances = dict(zip(FEATURES, [round(v*100, 1) for v in model.feature_importances_]))
correlations = {col: round(float(df[col].corr(df["grades"])), 3)
                for col in ["study_hours", "gaming_hours", "addiction_score",
                            "device_usage", "reaction_time_ms", "sleep_hours"]}

sample_idx  = np.linspace(0, len(y_test)-1, 60, dtype=int)
y_test_arr  = y_test.values
scatter_pts = [{"x": round(float(y_test_arr[i]), 1),
                "y": round(float(y_pred[i]), 1)} for i in sample_idx]

GENDERS = list(le_gender.classes_)
GENRES  = list(le_genre.classes_)
STRESS  = list(le_stress.classes_)

DASHBOARD = {
    "n_students": len(df),
    "avg_grade":  round(float(df["grades"].mean()), 1),
    "avg_gaming": round(float(df["gaming_hours"].mean()), 1),
    "avg_study":  round(float(df["study_hours"].mean()), 1),
    "r2": R2, "rmse": RMSE, "mae": MAE,
    "importances": importances,
    "correlations": correlations,
    "scatter": scatter_pts,
    "genders": GENDERS,
    "genres":  GENRES,
    "stress":  STRESS,
}

# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", data=json.dumps(DASHBOARD))


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json()
    try:
        row = np.array([[
            float(body["gaming_hours"]),
            float(body["study_hours"]),
            float(body["sleep_hours"]),
            float(body["attendance"]),
            float(body["social_activity"]),
            float(body["device_usage"]),
            float(body["reaction_time_ms"]),
            float(body["addiction_score"]),
            int(le_gender.transform([body["gender"]])[0]),
            int(le_genre.transform([body["gaming_genre"]])[0]),
            int(le_stress.transform([body["stress_level"]])[0]),
            int(body["age"]),
        ]])
        row_scaled = scaler.transform(row)
        pred = float(model.predict(row_scaled)[0])
        pred = max(0, min(pred, 100))

        if pred >= 75:   risk = "Bajo"
        elif pred >= 50: risk = "Medio"
        else:            risk = "Alto"

        return jsonify({"grade": round(pred, 1), "risk": risk})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
