import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
spam_classifier = pickle.load(open('spam_classifier.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict_api', methods=['POST'])
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']['email']  
    print(data)
    vectorized_input = vectorizer.transform(data)  
    prediction = spam_classifier.predict(vectorized_input)
    print(prediction.tolist())  
    return jsonify({'prediction': prediction.tolist()})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['email']
    final_input = vectorizer.transform([data])
    print(final_input)
    output = spam_classifier.predict(final_input).tolist()
    return render_template('index.html', prediction_text='The email is {}'.format('Spam' if output[0] == 1 else 'Not Spam'))


if __name__ == "__main__":
    app.run(debug=True)

