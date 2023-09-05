import gradio as gr
import joblib as jb
import numpy as np
import pandas as pd

input_types = ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']

def predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):

    model = jb.load('model.pkl')

    Pregnancies = int(Pregnancies)
    Glucose = float(Glucose)
    BloodPressure = float(BloodPressure)
    SkinThickness = float(SkinThickness)
    Insulin = float(Insulin)
    BMI = float(BMI)
    DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
    Age = int(Age)

    X = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]], 
                     columns=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"])
    
    p = model.predict_proba(X)[0]
    return {'Positive':p[0], 'Negative':p[1]}

demo = gr.Interface(fn=predict, inputs=input_types, outputs='label')
demo.launch()
