#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 02:33:49 2021

@author: anik
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle


app=Flask(__name__)

# load the classifier.pkl
pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)

# set loading path
@app.route('/')
def welcome():
    return "hello world"

# API: prediction from the independent variables
@app.route('/predict')
def predict_note_authentication():
    variance=request.args.get('variance')
    skewness=request.args.get('skewness')
    curtosis=request.args.get('curtosis')
    entropy=request.args.get('entropy')
    prediction=classifier.predict([[variance, skewness, curtosis, entropy]])
    return "The predicted value is"+ str(prediction)

# API: prediction from the csv file (independent variables)
@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return "The predicted values for the csv are"+ str(list(prediction))


if __name__ == "__main__":
    app.run()
    
    
    
    
    
    
    
    