import streamlit as st
from flask import Flask, request, jsonify, json
import pickle
import numpy as np


def load_model():
    with open('gliris.pkl','rb') as file:
        data = pickle.load(file=file)
    return data
model = load_model()

def show_predict_page():
    st.title("Iris data class prediction")
    st.write("""### We need some info to make class prediction""")
    # features = ("Sepal length","Sepal Width", "Petal Length", "Petal Width")
    sl = st.number_input('Sepal_Length')
    sw = st.number_input('Sepal_Width')
    pl = st.number_input('Petal_Length')
    pw = st.number_input('Petal_Width')
    
    ok = st.button("Classify the plants")
    if ok:
        # Make prediction using model loaded from disk as per the data.
        predict_request=[[sl, sw, pl, pw]]
        predict_request=np.array(predict_request)
        prediction = model.predict(predict_request)
        print(prediction)
        # Take the first value of prediction
        output = prediction[0]
        st.write("The predicted class is: ", output)                
    
        
        
    