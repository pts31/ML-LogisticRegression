

import numpy as np
import pandas as pd #not of your use
import logging
import json

FILE_NAME_TRAIN = 'train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'test.csv' #replace
ALPHA = 1e-3
EPOCHS = 100000
MODEL_FILE = 'models/model1'
train_flag = True

logging.basicConfig(filename='output.log',level=logging.DEBUG)


#utility functions
def loadData(file_name):
    df = pd.read_csv(file_name)
    logging.info("Number of data points in the data set "+str(len(df)))
    y_df = df['output']
    keys = ['company_rating','model_rating', 'bought_at', 'months_used', 'issues_rating','resale_value']
    X_df = df.get(keys)
    return X_df, y_df


def normalizeData(X_df, y_df, model):
    #save the scaling factors so that after prediction the value can be again rescaled
    model['input_scaling_factors'] = [list(X_df.mean()),list(X_df.std())]
   # model['output_scaling_factors'] = [y_df.mean(), y_df.std()]
    X = np.array((X_df-X_df.mean())/X_df.std())
    #y = np.array((y_df - y_df.mean())/y_df.std())
    return X, y_df , model

def normalizeTestData(X_df, y_df, model):
    meanX = model['input_scaling_factors'][0]
    stdX = model['input_scaling_factors'][1]
    #meany = model['output_scaling_factors'][0]
    #stdy = model['output_scaling_factors'][1]

    X = 1.0*(X_df - meanX)/stdX
    #y = 1.0*(y_df - meany)/stdy

    return X, y_df


def accuracy(X, y, model):

    y_predicted = predict(X,np.array(model['theta']))
    #acc = np.sqrt(1.0*(np.sum(np.square(y_predicted - y)))/len(X))
    l=len(y_predicted)
    n=0
    for i in range (l):
        if y_predicted[i]>=0.5:
            y_predicted[i]=1
        else:
            y_predicted[i]=0
        if y_predicted[i]==y[i]:
            n=n+1 
    acc=(1.0*n)/l*100
    #print "error associated with thi model is "+str(acc)
    return acc

def predict(X,theta):
    return np.dot(X,theta)