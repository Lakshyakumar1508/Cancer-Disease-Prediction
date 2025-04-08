import numpy as np
import pandas as pd
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler

# Load dataset once
dataset = pd.read_csv("The_Cancer_data_1500_V2.csv")

# Preprocess data
X = dataset.iloc[:, :-1]
y = dataset["Diagnosis"]

# Handle class imbalance
ros = RandomUnderSampler()
X_resampled, y_resampled = ros.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, train_size=0.8, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model once
model = LogisticRegression(multi_class="multinomial")
model.fit(X_train, y_train)

def home(request):
    return render(request, "home.html")

def prediction(request):
    return render(request, "prediction.html")

def result(request):
    try:
        input_values = [float(request.GET[f'n{i}']) for i in range(1, 9)]
    except (KeyError, ValueError):
        return render(request, "prediction.html", {"result2": "Invalid input. Please provide all values as numbers."})

    # Normalize input using the same scaler
    input_array = scaler.transform([input_values])

    # Predict
    prediction = model.predict(input_array)[0]
    result_text = "Our analysis suggests that the predicted cancer status is: \n" + ("Positive. Please consult a medical professional for further evaluation." if prediction == 1 else "Negative. However, regular check-ups are always recommended.")


    return render(request, "prediction.html", {"result2": result_text})
