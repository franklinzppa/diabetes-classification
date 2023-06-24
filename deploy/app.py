import gradio as gr
import joblib as jb

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
    
    p = model.predict_proba([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])[0]
    return {'Positive':p[0], 'Negative':p[1]}


input = ['number', 'number', 'number', 'number', 'number', 'number', 'number', 'number']

demo = gr.Interface(fn=predict, inputs=input, outputs='label')
demo.launch()
