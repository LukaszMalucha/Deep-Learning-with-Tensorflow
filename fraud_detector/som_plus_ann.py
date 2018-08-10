### Self Organizing Map - Fraud Detector

### Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


### Importing dataset http://archive.ics.uci.edu/ml/datasets/statlog+(australian+credit+approval)

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values                   # 1 - bank client application was approved, 0 - was rejected

### Feature Scaling for X (client features)

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))        # range 0-1 for normalization
X = sc.fit_transform(X)


# Training Self Organizing Map

from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.01)  # x,y for grid dimms ; input_len for x attributes; sigma default; l_r default
som.random_weights_init(X)                          # initialize weights with random small number
som.train_random(data = X, num_iteration = 100)


# Visualizing the results (higher mid, higher chance for outlayer - fraud)
from pylab import bone, pcolor, colorbar, plot, show
## Map of SOM nodes
bone()                                ## window for a map
pcolor(som.distance_map().T)          ## matrix of all the node distances/different colors (T for transpose), 
colorbar()                            ## color legend 

## Customers
markers = ['o', 's']                  ## circles & squares
colors = ['r', 'g']                   ## colors
for i, x in enumerate(X):             ## i for row, x for customer attributes in form of vector 
    w = som.winner(x)                 ## returns winning node for customer x
    plot(w[0] + 0.5,                  ## plot the marker into center of the winning node (0.5 to move to the middle)
         w[1] + 0.5,                  ## y-cord
         markers[y[i]],               ## did customer got approval class
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',    ## no inside marker color for better visibilty 
         markersize = 10,
         markeredgewidth = 2)
show()


### Identify Frauds

mappings = som.win_map(X)   ## map coordinates of the winning nodes (with associated customers)
frauds = np.concatenate((mappings[(7,7)], mappings[(3,5)]), axis = 0)        ## coords of white squares
frauds = sc.inverse_transform(frauds)                                        ## inversing normalization 




### Creating features matrix for ann

customers = dataset.iloc[:, 1:].values

### Create label and replace 0 with 1 for potentially fraudalent customers

is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
        if dataset.iloc[i,0] in frauds:
                is_fraud[i] = 1
        
##### TRAIN ANN to detect fraud

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
customers = sc.fit_transform(customers)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model     # saving model

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))

# Adding the second hidden layer
classifier.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size = 10, epochs = 5)

classifier.save('fraud_classifier.h5')     


# Part 3 - Making predictions and evaluating the model

# Predicting the fraud probability
fraud_prob = classifier.predict(customers)


fraud_prob = np.concatenate((dataset.iloc[:,0:1].values, fraud_prob), axis = 1)  
fraud_prob = fraud_prob[fraud_prob[:,1].argsort()]
