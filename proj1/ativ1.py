# ROAD MAP
# DESCONSIDERAR A PRIMEIRA COLUNA
# SEPARA A COLUNA DE PRECO DAS DEMAIS
# FAZER A MULTIPLICACAO DA MATRIX EM SEGUIDA DA INVERSA

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from enum import Enum
import numpy as np
import csv

CAR_AVG = (0.2 + 5.01)/2
CAR_MAX = 5.01
X_AVG = (0 + 10.74)/2
X_MAX = 10.74
Y_AVG = (0 + 58.9)/2
Y_MAX = 58.9
Z_AVG = (0 + 31.8)/2
Z_MAX = 31.8
DEP_AVG = (43 + 79)/2
DEP_MAX = 79
TAB_AVG = (43 + 95)/2
TAB_MAX = 95

def classificationCut (data):
    num_rows = len(data)
    for i in range(num_rows):
        if (data[i][1] == 'Fair'):
            data[i][1] = -2
        elif (data[i][1] == 'Good'):
            data[i][1] = -1
        elif (data[i][1] == 'Very Good'):
            data[i][1] = 0
        elif (data[i][1] == 'Premium'):
            data[i][1] = 1
        elif (data[i][1] == 'Ideal'):
            data[i][1] = 2

def classificationColor (data):
    num_rows = len(data)
    for i in range(num_rows):
        if (data[i][2] == 'D'):
            data[i][2] = -3
        elif (data[i][2] == 'E'):
            data[i][2] = -2
        elif (data[i][2] == 'F'):
            data[i][2] = -1
        elif (data[i][2] == 'G'):
            data[i][2] = 0
        elif (data[i][2] == 'H'):
            data[i][2] = 1
        elif (data[i][2] == 'I'):
            data[i][2] = 2
        elif (data[i][2] == 'J'):
            data[i][2] = 3

#como fazer com numeros pares?
def classificationClarity (data):
    num_rows = len(data)
    for i in range(num_rows):
        if (data[i][3] == 'I1'):
            data[i][3] = -4
        elif (data[i][3] == 'SI2'):
            data[i][3] = -3
        elif (data[i][3] == 'SI1'):
            data[i][3] = -2
        elif (data[i][3] == 'VS2'):
            data[i][3] = -1
        elif (data[i][3] == 'VS1'):
            data[i][3] = 1
        elif (data[i][3] == 'VVS2'):
            data[i][3] = 2
        elif (data[i][3] == 'VVS1'):
            data[i][3] = 3
        elif (data[i][3] == 'IF'):
            data[i][3] = 4

def classificationSet (data):
    classificationCut(data)
    classificationColor(data)
    classificationClarity(data)

# Main function

def formatArray (array) :
    withouHeader = np.asarray(array[1:])
    target = withouHeader[:,9]
    params = np.delete(withouHeader, 9, axis = 1)
    classificationSet(params)
    params = params.astype(float)
    target = map(float, target)
    return params, target

def findNormalParams (fullParams, fullTarget):
    diamondsFullParamsTranspose = fullParams.transpose()
    multiplyMatrix = np.matmul (diamondsFullParamsTranspose, fullParams)
    inverseMatrix = np.linalg.inv(multiplyMatrix)
    multiplyInverseTranspose = np.matmul(inverseMatrix, diamondsFullParamsTranspose)
    paramsValues = np.matmul(multiplyInverseTranspose, fullTarget)
    return paramsValues

def fitParams (array):
    scaler = MinMaxScaler(feature_range=(-0.5,0.5))
    scaler.fit(array)
    return scaler.transform(trainParams)


with open('diamonds-dataset/diamonds-train.csv', 'rb') as f:
    reader = csv.reader(f)
    diamondsTrain = list(reader)
with open('diamonds-dataset/diamonds-test.csv', 'rb') as f:
    reader = csv.reader(f)
    diamondsTest = list(reader)


testParams, testTarget = formatArray(diamondsTest)
trainParams, trainTarget = formatArray(diamondsTrain)
trainParamsFit = fitParams (trainParams)
testParamsFit = fitParams (testParams)
# scaler = MinMaxScaler()

fullParams = np.concatenate((testParams, trainParams), axis = 0)
fullTarget = np.concatenate((testTarget, trainTarget), axis = 0)

normalParams = findNormalParams (fullParams, fullTarget)

print (normalParams)
