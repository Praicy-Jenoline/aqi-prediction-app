#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import pandas as pd
import joblib
import shap
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from utils import classify_aqi

app = Flask(__name__)

# ---------------------------------------------------
# LOAD MODEL + FEATURES
# ---------------------------------------------------
model = joblib.load("model.pkl")
feature_names = joblib.load("features.pkl")

print("Loaded Model:", type(model).__name__)

# ---------------------------------------------------
# LOCATION MAP
# ---------------------------------------------------
location_map = {
    0: "Delhi",
    1: "Faridabad",
    2: "Ghaziabad",
    3: "Gurgaon",
    4: "Noida"
}

# ---------------------------------------------------
# SHAP EXPLAINER FOR LIGHTGBM
# ---------------------------------------------------
try:

    explainer = shap.TreeExplainer(model)

    print("SHAP Loaded Successfully")

except Exception as e:

    print("SHAP Error:", e)

    explainer = None

# ---------------------------------------------------
# FEATURES SHOWN IN UI
# ---------------------------------------------------
visible_features = [
    "location_name",
    "location_lat",
    "location_lon",
    "hour",
    "day",
    "month",
    "year",
    "day_of_week",
    "is_weekend",
    "pollution_index"
]

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------
@app.route('/')
def home():

    return render_template(
        "index.html",
        features=visible_features
    )

# ---------------------------------------------------
# PREDICT ROUTE
# ---------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():

    try:

        # ---------------------------------------------------
        # USER INPUT
        # ---------------------------------------------------
        input_data = {}

        for feature in visible_features:

            value = request.form.get(feature)

            if value is None or value == "":
                return f"Missing value for {feature}"

            input_data[feature] = float(value)

        # ---------------------------------------------------
        # CREATE TIMESTAMP FEATURES
        # ---------------------------------------------------
        input_data["ts_hour"] = input_data["hour"]
        input_data["ts_day"] = input_data["day"]
        input_data["ts_month"] = input_data["month"]
        input_data["ts_year"] = input_data["year"]

        # ---------------------------------------------------
        # FEATURE ORDER
        # ---------------------------------------------------
        final_input = {}

        for feature in feature_names:

            final_input[feature] = input_data[feature]

        # ---------------------------------------------------
        # DATAFRAME
        # ---------------------------------------------------
        input_df = pd.DataFrame([final_input])

        # ---------------------------------------------------
        # PREDICTION
        # ---------------------------------------------------
        prediction = model.predict(input_df)[0]

        # ---------------------------------------------------
        # AQI CATEGORY + ADVICE
        # ---------------------------------------------------
        category, advice = classify_aqi(prediction)

        # ---------------------------------------------------
        # LOCATION NAME
        # ---------------------------------------------------
        location_name = location_map.get(
            int(input_data["location_name"]),
            "Unknown"
        )

        # ---------------------------------------------------
        # CLEAN REPORT
        # ---------------------------------------------------
        report = {

            "Location": location_name,

            "Coordinates":
                f"{input_data['location_lat']}, "
                f"{input_data['location_lon']}",

            "Date":
                f"{int(input_data['day'])}/"
                f"{int(input_data['month'])}/"
                f"{int(input_data['year'])}",

            "Hour":
                int(input_data["hour"]),

            "Weekend":
                "Yes" if input_data["is_weekend"] == 1 else "No",

            "Pollution Index":
                round(input_data["pollution_index"], 2)
        }

        # ---------------------------------------------------
        # REMOVE OLD SHAP IMAGE
        # ---------------------------------------------------
        shap_path = "static/shap_single.png"

        if os.path.exists(shap_path):

            os.remove(shap_path)

        # ---------------------------------------------------
        # GENERATE DYNAMIC SHAP
        # ---------------------------------------------------
        shap_generated = False

        if explainer is not None:

            try:

                # -------------------------------------------
                # SHAP VALUES
                # -------------------------------------------
                shap_values = explainer.shap_values(input_df)

                # -------------------------------------------
                # CREATE FIGURE
                # -------------------------------------------
                plt.figure(figsize=(14, 7))

                # -------------------------------------------
                # MODERN SHAP WATERFALL
                # -------------------------------------------
                shap.plots.waterfall(

                    shap.Explanation(
                        values=shap_values[0],
                        base_values=explainer.expected_value,
                        data=input_df.iloc[0],
                        feature_names=feature_names
                    ),

                    show=False
                )

                # -------------------------------------------
                # SAVE IMAGE
                # -------------------------------------------
                plt.tight_layout()

                plt.savefig(
                    shap_path,
                    bbox_inches='tight',
                    dpi=300
                )

                plt.close()

                shap_generated = True

                print("SHAP graph generated successfully")

            except Exception as shap_error:

                print("SHAP Generation Error:", shap_error)

        # ---------------------------------------------------
        # RETURN RESULT
        # ---------------------------------------------------
        return render_template(
            "index.html",
            prediction=round(prediction, 2),
            category=category,
            advice=advice,
            report=report,
            shap_generated=shap_generated,
            features=visible_features
        )

    except Exception as e:

        return f"Error: {str(e)}"

# ---------------------------------------------------
# RUN APP
# ---------------------------------------------------
if __name__ == "__main__":

    app.run(debug=True)

