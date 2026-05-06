#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
import joblib
import numpy as np
from utils import classify_aqi

app = Flask(__name__)

model = joblib.load("model.pkl")

# Color mapping
def get_color(category):
    colors = {
        "Good": "green",
        "Satisfactory": "lightgreen",
        "Moderate": "orange",
        "Poor": "red",
        "Very Poor": "darkred",
        "Severe": "purple"
    }
    return colors.get(category, "black")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Store form inputs
        form_data = {
            'pm25': request.form['pm25'],
            'pm10': request.form['pm10'],
            'no2': request.form['no2'],
            'co': request.form['co'],
            'ozone': request.form['ozone'],
            'month': request.form['month'],
            'day': request.form['day'],
            'holiday': request.form['holiday']
        }

        data = np.array([[
            float(form_data['pm25']),
            float(form_data['pm10']),
            float(form_data['no2']),
            float(form_data['co']),
            float(form_data['ozone']),
            float(form_data['month']),
            float(form_data['day']),
            float(form_data['holiday'])
        ]])

        prediction = model.predict(data)[0]
        category = classify_aqi(prediction)
        color = get_color(category)

        # Simple health message
        if category in ["Good", "Satisfactory"]:
            advice = "Air quality is acceptable. Minimal impact on health."
        elif category in ["Moderate", "Poor"]:
            advice = "Sensitive individuals should reduce outdoor activity."
        else:
            advice = "Health alert: Avoid outdoor exposure. Wear masks."

        return render_template(
            'index.html',
            prediction=round(prediction, 2),
            category=category,
            color=color,
            advice=advice,
            form_data=form_data
        )

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == "__main__":
    app.run(debug=True)

