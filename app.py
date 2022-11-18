# Import libraries
import numpy as np
from flask import Flask, request, jsonify, json
import pickle
import os
# Load the model
model = pickle.load(open('gliris.pkl','rb'))
app = Flask(__name__)
@app.route('/api',methods=['POST'])
def predict():
    data = request.json['data']
    # Make prediction using model loaded from disk as per the data.
    predict_request=[[data['sl'],data['sw'],data['pl'],data['pw']]]
    predict_request=np.array(predict_request)
    print(predict_request)
    prediction = model.predict(predict_request)
    print(prediction)
    # Take the first value of prediction
    output = prediction[0]
    print(output)
    return jsonify(int(output))

if __name__ == '__main__':
    app.run(debug=True)