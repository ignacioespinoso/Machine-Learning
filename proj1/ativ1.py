# ROAD MAP
# DESCONSIDERAR A PRIMEIRA COLUNA
# SEPARA A COLUNA DE PRECO DAS DEMAIS
# FAZER A MULTIPLICACAO DA MATRIX EM SEGUIDA DA INVERSA

from sklearn.model_selection import train_test_split
from enum import Enum
import numpy as np
import csv

def classificationCut (data):
    num_rows = len(data)
    for i in range(num_rows):
        if (data[i][2] == 'Fair'):
            data[i][2] = -2
        elif (data[i][2] == 'Good'):
            data[i][2] = -1
        elif (data[i][2] == 'Very Good'):
            data[i][2] = 0
        elif (data[i][2] == 'Premium'):
            data[i][2] = 1
        elif (data[i][2] == 'Ideal'):
            data[i][2] = 2

def classificationColor (data):
    num_rows = len(data)
    for i in range(num_rows):
        if (data[i][3] == 'D'):
            data[i][3] = -3
        elif (data[i][3] == 'E'):
            data[i][3] = -2
        elif (data[i][3] == 'F'):
            data[i][3] = -1
        elif (data[i][3] == 'G'):
            data[i][3] = 0
        elif (data[i][3] == 'H'):
            data[i][3] = 1
        elif (data[i][3] == 'I'):
            data[i][3] = 2
        elif (data[i][3] == 'J'):
            data[i][3] = 3

#como fazer com numeros pares?
def classificationClarity (data):
    num_rows = len(data)
    for i in range(num_rows):
        if (data[i][4] == 'I1'):
            data[i][4] = -4
        elif (data[i][4] == 'SI2'):
            data[i][4] = -3
        elif (data[i][4] == 'SI1'):
            data[i][4] = -2
        elif (data[i][4] == 'VS2'):
            data[i][4] = -1
        elif (data[i][4] == 'VS1'):
            data[i][4] = 1
        elif (data[i][4] == 'VVS2'):
            data[i][4] = 2
        elif (data[i][4] == 'VVS1'):
            data[i][4] = 3
        elif (data[i][4] == 'IF'):
            data[i][4] = 4

def classificationSet (data):
    classificationCut(data)
    classificationColor(data)
    classificationClarity(data)

# Main function

with open('diamonds.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)

asNumpy = np.asarray(your_list[1:45850])

classificationSet(asNumpy)
print(asNumpy[:20])

floatMatrix = asNumpy.astype(float)
print(floatMatrix[:5])

print(np.mean(floatMatrix))
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
# print(X_train)
