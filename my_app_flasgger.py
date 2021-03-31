#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 04:44:25 2021

@author: anik
"""


from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
# for the front end
import flasgger
from flasgger import Swagger

app=Flask(__name__)
# for the front end
Swagger(app)

# load the classifier.pkl
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

# set loading path
@app.route('/')
def welcome():
    return "hello world"

# API: prediction from the independent variables
@app.route('/predict', methods=['Get'])
def predict_note_authentication():
    
    
    """Let's authenticate the Bank Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
      200:
        description: The output values
    """
    
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance, skewness, curtosis, entropy]])
    return str(prediction)

# API: prediction from the csv file (independent variables)
@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    
    """Let's authenticate the Bank Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
      200:
        description: The output values
    """
    
    
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return str(list(prediction))


if __name__ == "__main__":
    app.run()
    
    
    
    
    
    
    
    