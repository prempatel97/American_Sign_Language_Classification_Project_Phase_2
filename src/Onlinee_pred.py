#!/usr/bin/python3.6

"""
This file takes the json input and uses the saved machine learning models to predict the class of the sign
"""

from flask import Flask, jsonify
from flask import request
import json
import numpy as np
import pandas as pd
import pickle
import os
import csv
import stat
from sklearn import preprocessing
from collections import Counter

app = Flask(__name__)

def center_by_nose(df):
    dfX = df.filter(regex='_x') #all X values
    dfY = df.filter(regex='_y') #all Y values
    dfX = dfX.sub(dfX["nose_x"], axis=0)
    dfY = dfY.sub(dfY["nose_y"], axis=0)
    return dfX, dfY

def normalize_dataframe(df):
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

@app.route('/postjson', methods = ['POST'])
def postJsonHandler():
    print(request.is_json)
    content = request.get_json()
    print(content)
    result1 = ""
    result2 = ""
    result3 = ""
    result4 = ""
    dict1 = content

    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']

    csv_data = np.zeros((len(dict1), len(columns)))
    for i in range(csv_data.shape[0]):
        one = []
        one.append(dict1[i]['score'])
        for obj in dict1[i]['keypoints']:
            one.append(obj['score'])
            one.append(obj['position']['x'])
            one.append(obj['position']['y'])
        csv_data[i] = np.array(one)
    df = pd.DataFrame(csv_data, columns=columns)

    #Drop un-neccessary key-points
    df.drop(columns=["score_overall", "nose_score", "leftEye_score", "rightEye_score", "leftEar_score","rightEar_score", "leftShoulder_score", "rightShoulder_score", "leftElbow_score", "rightElbow_score", "leftWrist_score","rightWrist_score","leftHip_score","rightHip_score",'leftKnee_score', 'leftKnee_x', 'leftKnee_y','rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y','rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y', 'leftEye_x', 'leftEye_y', 'rightEye_x', 'rightEye_y'], inplace=True)

    dfX, dfY = center_by_nose(df)
    dfX= dfX.div(dfY['rightHip_y'],axis=0)
    dfY = dfY.div(dfY['rightHip_y'], axis=0)
    # Create x, where x the 'scores' column's values as floats
    dfX=normalize_dataframe(dfX)
    dfY=normalize_dataframe(dfY)

    df = pd.concat([dfX,dfY], axis=1)


    X_test = df

    loaded_model = pickle.load(open('forest.sav', 'rb'))
    result1 = loaded_model.predict(X_test)
    res1 = Counter(result1).most_common(1)

    loaded_model = pickle.load(open('svm.sav', 'rb'))
    result2 = loaded_model.predict(X_test)
    res2 = Counter(result2).most_common(1)

    loaded_model = pickle.load(open('mlp.sav', 'rb'))
    result3 = loaded_model.predict(X_test)
    res3 = Counter(result3).most_common(1)

    loaded_model = pickle.load(open('knn.sav', 'rb'))
    result4 = loaded_model.predict(X_test)
    res4 = Counter(result4).most_common(1)

    data = {str("1"): str(res1[0][0]), str("2"): str(res2[0][0]),
            str("3"): str(res3[0][0]), str("4"): str(res4[0][0])}

    return jsonify(data)

app.run(host='172.31.34.169', port=8073)
