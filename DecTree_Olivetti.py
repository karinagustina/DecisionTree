#Decision Tree with Sklearn Datasets - Olivetti Faces

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#================================================
#Load Data
#================================================

from sklearn.datasets import fetch_olivetti_faces
face = fetch_olivetti_faces()
print(dir(face))
print(len(face['data'][0]))        #64*64 = 4096
print(face['images'][0])           #64 array @64 elements
print(face['target'])              #40 person @10 poses

#================================================
#Plot Faces
#================================================

fig = plt.figure('Olivetti Faces', figsize = (10,4))
for i in range(10):
    person = 29              #start from person[0]; ended by person[39]
    plt.subplot(2,5,i+1)
    plt.imshow(face['images'][i + (10 * person)], cmap = 'gray')
    plt.suptitle('Faces of Person No.{}'.format(person))

plt.show()

#================================================
#Decision Tree Algorithm
#================================================

from sklearn.model_selection import train_test_split

#Split Train (90%) and Test (10%)
x_train, x_test, y_train, y_test = train_test_split(
    face['data'],
    face['target'],
    test_size = .1
)
print(len(x_train))
print(len(x_test))

#Create Decision Tree Model
from sklearn import tree
model = tree.DecisionTreeClassifier()

#Training Model
model.fit(x_train, y_train)

#Testing Model Accuracy
print(model.score(x_train, y_train))

#Prediction
print(x_test[0])
print(model.predict([x_test[0]]))
print(y_test[0])

#================================================
#Plot Decision Tree Prediction
#================================================

plt.figure('Olivetti Faces Prediction', figsize = (4, 4))
plt.imshow(x_test[0].reshape(64, 64), cmap = 'gray')
plt.title('Actual: {} / Prediction: {}'.format(
    y_test[0],
    model.predict([x_test[0]])[0]
))

plt.show()
