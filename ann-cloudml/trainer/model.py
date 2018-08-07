## Artificial Neural Network

# Installing Theano - fast numerical computation (runs on cpu and gpu)
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow - for fast computation (installation cause an issues atm)
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras - wraps both above libraries
# pip install --upgrade keras


########################################################### DATA PREPROCESSING ######################################################


# Importing the libraries
import numpy as np
import pandas as pd
import os

from tensorflow.python.lib.io import file_io
from pandas.compat import StringIO

def read_data(gcs_path):
   print('downloading csv file from', gcs_path)     
   file_stream = file_io.FileIO(gcs_path, mode='r')
   data = pd.read_csv(StringIO(file_stream.read()))
   print(data.head())
   return data

# Importing the dataset
dataset = read_data('gs://churn-modelling-mlengine/data/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values   ## Everything except row number, Name and CustomerId
y = dataset.iloc[:, 13].values     ## Customer Exited 0-1

# Encoding categorical data into 0-1 values (text to numbers)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])    ## transform country variable
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])    ## transform gender variable
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]                         ## remove first column to avoid dummy variable trap !!!!!!!!!!!!!!!!

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#################################################### ARTIFICIAL NEURAL NETWORK ######################################################

## Import Keras libraries and packages
import keras
from keras.models import Sequential          ## to initialise neuro-network
from keras.layers import Dense               ## to create layers
from keras.models import load_model          ## saving ann


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
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=1)

classifier.save('classifier.h5') 

with file_io.FileIO('classifier.h5', mode='rb') as input_f:
    with file_io.FileIO('gs://churn-modelling-mlengine/data/classifier.h5', mode='wb+') as output_f:
        output_f.write(input_f.read())
