# ROAD MAP
# Terminar de testar o sklearning e criar nossa propria funcao
#     funcao de calcular o h_teta
#     funcao para recalcular o teta de cada uma entrada do array

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from enum import Enum
import numpy as np
import csv

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

# Transforma as strings em numeros
def classificationSet (data):
    classificationCut(data)
    classificationColor(data)
    classificationClarity(data)

# Retira o header, separa em parametros e target, formata em float e retorna em formato Array
def formatArray (array) :
    withouHeader = np.asarray(array[1:])
    target = withouHeader[:,9]
    params = np.delete(withouHeader, 9, axis = 1)
    classificationSet(params)
    params = params.astype(float)
    target = map(float, target)
    return params, target

# Realiza as multiplicacoes de matrizes e retorna um array com os parametros
def findNormalParams (fullParams, fullTarget):
    diamondsFullParamsTranspose = fullParams.transpose()
    multiplyMatrix = np.matmul (diamondsFullParamsTranspose, fullParams)
    inverseMatrix = np.linalg.inv(multiplyMatrix)
    multiplyInverseTranspose = np.matmul(inverseMatrix, diamondsFullParamsTranspose)
    paramsValues = np.matmul(multiplyInverseTranspose, fullTarget)
    return paramsValues

# Transforma os parametros em valores entre -0.5 e 0.5
def fitParams (array):
    scaler = MinMaxScaler(feature_range=(-0.5,0.5))
    scaler.fit(array)
    return scaler.transform(trainParams)


# Main function

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

fullParams = np.concatenate((testParams, trainParams), axis = 0)
fullTarget = np.concatenate((testTarget, trainTarget), axis = 0)

normalParams = findNormalParams (fullParams, fullTarget)

print (normalParams)
