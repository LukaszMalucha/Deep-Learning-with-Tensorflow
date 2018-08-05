import argparse

from keras.callbacks import LambdaCallback

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import keras
from keras.models import Sequential          ## to initialise neuro-network
from keras.layers import Dense               ## to create layers
from keras.models import load_model          ## saving ann


def saveModelToCloud(epoch, period=1):
        if epoch % period == 0:
                saver.saveModelToCloud(model, pathToJobDir + '/epochs' + jobName, '{:03d}'.format(epoch))
                
                
if __name__ == '__main__':
        
        parser = argparse.ArgumentParser() 
        parser.add_argument(
                        '--train-file',
                        help='GCS or local paths to training data',
                        required=True
        )

        parser.add_argument(
                        '--job-name',
                        help='GCS to write checkpoints and export models',
                        required=True
        )               
        
        parser.add_argument(
                        '--job-dir',
                        help='GCS to write checkpoints and export models',
                        required=True                        
                        )
        
        args = parser.parse_args()
        arguments = args.__dict__
       
        pathToJobDir = arguments.pop('job_dir')
        jobName = arguments.pop('job_name')
        pathToData = arguments.pop('train_file')
       
        ## MODEL

        # Importing the dataset
        dataset = pd.read_csv('Churn_Modelling.csv')
        X = dataset.iloc[:, 3:13].values   ## Everything except row number, Name and CustomerId
        y = dataset.iloc[:, 13].values     ## Customer Exited 0-1
        
        # Encoding categorical data into 0-1 values (text to numbers)
        
        labelencoder_X_1 = LabelEncoder()
        X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])    ## transform country variable
        labelencoder_X_2 = LabelEncoder()
        X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])    ## transform gender variable
        onehotencoder = OneHotEncoder(categorical_features = [1])
        X = onehotencoder.fit_transform(X).toarray()
        X = X[:, 1:]                         ## remove first column to avoid dummy variable trap !!!!!!!!!!!!!!!!
        
        # Splitting the dataset into the Training set and Test set
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        # Feature Scaling
        
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)
        
        ## Initialise the Artificial Neural Network
        classifier = Sequential()
        
        # Adding the input layer and the first hidden layer - best practice - avereage between input nodes and output nodes((11+1)/2)
        classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim=11))  ### relu for rectifier (hidden node)
        
        ## Adding the second hidden layer
        
        classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))
        
        ## Adding the output layer  
        classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))  ## only one node(yes/no) and activation function changed to sigmoid
                                                                                         ## if output variable is encoded then dim=3, activation = softmax
        
        ## Compiling the Artificial Neural Network - finding the best weight values with stochastic approach
        classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )    ## logarithmic loss
        
        ## Fitting the ANN to the Training set. Two additional arguments - batch size & number of epochs
        classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)
        
        ### Accuracy of 86% ###
        
        
        saver.saveModelToCloud(classifier, pathToJobDir)
       
       
       
       
       
       
       
       
       