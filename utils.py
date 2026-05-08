#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def classify_aqi(aqi):
    if aqi <= 50:
        return "Good", "Air quality is satisfactory. Minimal impact."
    elif aqi <= 100:
        return "Satisfactory", "Minor breathing discomfort for sensitive people."
    elif aqi <= 200:
        return "Moderate", "Breathing discomfort for people with lung disease."
    elif aqi <= 300:
        return "Poor", "Breathing discomfort for most people on prolonged exposure."
    elif aqi <= 400:
        return "Very Poor", "Respiratory illness on prolonged exposure."
    else:
        return "Severe", "Serious health effects. Avoid outdoor activities."

