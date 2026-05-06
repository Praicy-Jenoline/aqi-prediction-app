#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def classify_aqi(aqi):
    try:
        aqi = float(aqi)
    except:
        return "Invalid AQI"

    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

