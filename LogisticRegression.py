
import numpy as np
import logging
import json
#import math
from utility import * #custom methods for data cleaning

FILE_NAME_TRAIN = 'train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'test.csv' #replace
ALPHA = 3
EPOCHS = 5000#keep this greater than or equl to 5000 strictly otherwise you will get an error
MODEL_FILE = 'models/model1'
train_flag = True

logging.basicConfig(filename='output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)
#################################################################################################
#####################################write the functions here####################################
#################################################################################################
#this function appends 1 to the start of the input X and returns the new arrayxx
def appendIntercept(X):
    m=X.shape[0]
    col=np.ones((m,1))
    new_X=np.hstack((col,X))
    return new_X
    #steps
    #make a column vector of ones
    #stack this column vector infront of the main X vector using hstack
    #return the new matrix
    #remove this line once you finish writing




 #intitial guess of parameters (intialize all to zero)
 #this func takes the number of parameters that is to be fitted and returns a vector of zeros
def initialGuess(n_thetas):
    z=np.zeros(n_thetas)
    return z


def train(theta, X, y, model):
     J = [] #this array should contain the cost for every iteration so that you can visualize it later when you plot it vs the ith iteration
     #train for the number of epochs you have defined
     m = len(y)
     #your  gradient descent code goes here
     #steps
     #run you gd loop for EPOCHS that you have defined
        #calculate the predicted y using your current value of theta
        # calculate cost with that current theta using the costFunc function
        #append the above cost in J
        #calculate your gradients values using calcGradients function
        # update the theta using makeGradientUpdate function (don't make a new variable assign it back to theta that you received)
     for i in range (0,EPOCHS):
         yp=predict(X,theta)
         c=costFunc(m,y,yp)
         J.append(c)
         cg=calcGradients(X,y,yp,m)
         theta=makeGradientUpdate(theta,cg)
                 
     model['J'] = J
     model['theta'] = list(theta)
     return model


#this function will calculate the total cost and will return it
def costFunc(m,y,y_predicted):
    #takes three parameter as the input m(#training examples), (labeled y), (predicted y)
    #steps
    #apply the formula learnt
    #yd=np.subtract(y_predicted,y)
    #ys=np.multiply(yd,yd)
    #s=np.sum(ys)
    #j=s/(2*m)
    l = y*np.log(y_predicted) + (1 - y)*np.log(1 - y_predicted)
    J = np.sum(l)
    J = (-1.0)*(J/m)
    return J
    

def calcGradients(X,y,y_predicted,m):
    #apply the formula , this function will return cost with respect to the gradients
    # sically an numpy array containing n_params
    #yd=np.subtract(y_predicted,y)
    #l=X.shape[0]
    #yd=yd.reshape((l,1))
    #ym=np.multiply(yd,X)
    #ys=np.sum(ym,axis=0)
   # ys=np.dot(yd,X)
    #a=ys/m
    return np.dot((y_predicted-y),X)/m
    

#this function will update the theta and return it
def makeGradientUpdate(theta, grads):
    p=ALPHA*grads
    t=np.subtract(theta,p)
    return t
    

#this function will take two paramets as the input
def predict(X,theta):
    new=np.dot(X,theta.T)
    h=1.0/(1.0+np.exp(-1.0*new))
    return h

########################main function###########################################
def main():
    if(train_flag):
        model = {}
        X_df,y_df = loadData(FILE_NAME_TRAIN)
        X,y, model = normalizeData(X_df, y_df, model)
        X = appendIntercept(X)
        theta = initialGuess(X.shape[1])
        model = train(theta, X, y, model)
        #with open(MODEL_FILE,'w') as f:
         #   f.write(json.dumps(model))
        print "Accuracy of Training data ",accuracy(X,y,model)

    #else:
     #   model = {}
       # with open(MODEL_FILE,'r') as f:
        #    model = json.loads(f.read())
        X_df, y_df = loadData(FILE_NAME_TEST)
        X,y = normalizeTestData(X_df, y_df, model)
        X = appendIntercept(X)
        print "Accuracy of Test data ",accuracy(X,y,model)

if __name__ == '__main__':
    main()