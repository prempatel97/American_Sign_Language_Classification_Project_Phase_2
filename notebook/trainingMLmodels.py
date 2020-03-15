"""
This file trains 4 different machine learning models and saves the models to disk
"""

import json
import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from numpy import save
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import pickle

path_to_videos = "/home/dhruv/Allprojects/MC/Key_points_json_sir/"

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


def convert_to_csv(path_to_video):
    columns = ['score_overall', 'nose_score', 'nose_x', 'nose_y', 'leftEye_score', 'leftEye_x', 'leftEye_y',
               'rightEye_score', 'rightEye_x', 'rightEye_y', 'leftEar_score', 'leftEar_x', 'leftEar_y',
               'rightEar_score', 'rightEar_x', 'rightEar_y', 'leftShoulder_score', 'leftShoulder_x', 'leftShoulder_y',
               'rightShoulder_score', 'rightShoulder_x', 'rightShoulder_y', 'leftElbow_score', 'leftElbow_x',
               'leftElbow_y', 'rightElbow_score', 'rightElbow_x', 'rightElbow_y', 'leftWrist_score', 'leftWrist_x',
               'leftWrist_y', 'rightWrist_score', 'rightWrist_x', 'rightWrist_y', 'leftHip_score', 'leftHip_x',
               'leftHip_y', 'rightHip_score', 'rightHip_x', 'rightHip_y', 'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y']
    label = path_to_video.split("/")[-1].split("_")[0].lower()
    data = json.loads(open(path_to_video , 'r').read())

    csv_data = np.zeros((len(data), len(columns)))
    for i in range(csv_data.shape[0]):
        one = []
        one.append(data[i]['score'])
        for obj in data[i]['keypoints']:
            one.append(obj['score'])
            one.append(obj['position']['x'])
            one.append(obj['position']['y'])
        csv_data[i] = np.array(one)
    df = pd.DataFrame(csv_data, columns=columns)
    df.drop(columns=["score_overall", "nose_score", "leftEye_score", "rightEye_score", "leftEar_score","rightEar_score", "leftShoulder_score", "rightShoulder_score", "leftElbow_score", "rightElbow_score", "leftWrist_score"
      ,"rightWrist_score","leftHip_score","rightHip_score",'leftKnee_score', 'leftKnee_x', 'leftKnee_y',
               'rightKnee_score', 'rightKnee_x', 'rightKnee_y', 'leftAnkle_score', 'leftAnkle_x', 'leftAnkle_y',
               'rightAnkle_score', 'rightAnkle_x', 'rightAnkle_y', 'leftEye_x', 'leftEye_y', 'rightEye_x', 'rightEye_y'], inplace=True)

    dfX, dfY = center_by_nose(df)
    dfX= dfX.div(dfY['rightHip_y'],axis=0)
    dfY = dfY.div(dfY['rightHip_y'], axis=0)
    
    # Create x, where x the 'scores' column's values as floats
    dfX=normalize_dataframe(dfX)
    dfY=normalize_dataframe(dfY)

    df = pd.concat([dfX,dfY], axis=1)
    df['label'] = label
    df = df.truncate(after=70)
    return df



if __name__ == '__main__':

    files = os.listdir(path_to_videos)
    label_dic = {"gift":1, "book":2, "bell":3, "total":4, "car":5, "movie":0}
    labels=[]
    dataset=pd.DataFrame()
    for file in files:
      new_path = path_to_videos + file
      df = convert_to_csv(new_path)
      df_flat = df
      
      dataset = pd.concat([dataset,df_flat])

    X_train = dataset
    Y_train = dataset["label"]
    X = pd.DataFrame(X_train.drop(columns="label"))
    print(X.to_numpy())
    Y = pd.DataFrame(Y_train)
    print(Y.to_numpy().flatten())
    Y = Y.to_numpy().flatten()
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.15)
    
    model = RandomForestClassifier(n_estimators=70)
    model.fit(train_X, train_Y)
    filename1 = "forest.sav"
    pickle.dump(model, open(filename1,'wb'))
    print(f'The training accuracy for random forest:  {model.score(test_X, test_Y)}')

    clf = MLPClassifier(max_iter=1000, early_stopping=False ,hidden_layer_sizes = (100,100,60,20), learning_rate = 'adaptive')
    clf.fit(train_X, train_Y) 
    filename3 = "mlp.sav"
    pickle.dump(clf, open(filename3,'wb'))
    print(f'The training accuracy for MLPClassifier:  {clf.score(test_X, test_Y)}')

    svm_model_linear = SVC(C=0.7, gamma = 'scale').fit(train_X, train_Y)
    filename2 = "svm.sav"
    pickle.dump(svm_model_linear, open(filename2,'wb'))
    print(f'The training accuracy for SVM:  {svm_model_linear.score(test_X, test_Y)}')

    knn = KNeighborsClassifier(n_neighbors=4).fit(train_X, train_Y)
    filename4 = "knn.sav"
    pickle.dump(knn, open(filename4,'wb'))
    print(f'The training accuracy for kNN:  {knn.score(test_X, test_Y)}')
