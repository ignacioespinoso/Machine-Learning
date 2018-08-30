# ROAD MAP
# Terminar de testar o sklearning e criar nossa propria funcao
#     funcao de calcular o h_teta
#     funcao para recalcular o teta de cada uma entrada do array

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
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

def addColumnTetaZero (array):
    return np.c_[np.ones(array.shape[0]), array]

def createArrayTheta (numberParams, multi):
    array = np.c_[np.ones(numberParams)]
    array = array * multi
    return array

# Main function

with open('diamonds-dataset/diamonds-train.csv', 'rb') as f:
    reader = csv.reader(f)
    diamondsTrain = list(reader)
with open('diamonds-dataset/diamonds-test.csv', 'rb') as f:
    reader = csv.reader(f)
    diamondsTest = list(reader)

testParams, testTarget = formatArray(diamondsTest)
trainParams, trainTarget = formatArray(diamondsTrain)

testParams = addColumnTetaZero(testParams)

thetas = createArrayTheta(10, 100)
# thetas = [[1],[1040],[170],[-320],[392],[-690],[77],[-208],[-9],[6]]
alpha = 0.01
m = testParams.shape[0]
print m
# FAZER O FOR ATE DETERMINADO NUMERO DE ITERACOES OU ERRO < X

testTargetTranspose = np.asarray(testTarget)
testTargetTranspose = testTargetTranspose.reshape(testTargetTranspose.shape[0], -1)



agaDeTheta = np.matmul (testParams, thetas)
preSomatorio = agaDeTheta - testTargetTranspose
somatorio = preSomatorio * testParams
somaTheta = np.sum(somatorio, axis=0)
somaTheta = somaTheta.reshape(somaTheta.shape[0], -1)
print(somaTheta.shape[1])
somaTheta = somaTheta / m
somaTheta = somaTheta * alpha
print (thetas)
thetas = thetas - somaTheta
print (thetas)







# SCIKIT-LEARN
# ADCIONAR A COLUNA DE 1

# alph = 0.1
# verb = True
# maxIter = 50000
# learningRate = "invscaling"
# eta = 0.01
# epsilo = 10
#
# linReg = linear_model.SGDRegressor(alpha= alph, average=False, epsilon=epsilo, eta0=eta,
#        fit_intercept=True, l1_ratio=0.15, learning_rate=learningRate,
#        loss='squared_loss', max_iter=maxIter, n_iter=None, penalty='l2',
#        power_t=0.25, random_state=None, shuffle=True, tol=None,
#        verbose=verb, warm_start=False)
#
# linReg.fit(trainParams, trainTarget)
# print (linReg.predict(testParams))
# print (linReg.score(trainParams, trainTarget))


# # NORMAL
# ADICIONAR A COLUNA DE 1 e analisar os resultados
# trainParamsFit = fitParams (trainParams)
# testParamsFit = fitParams (testParams)
#
# fullParams = np.concatenate((testParams, trainParams), axis = 0)
# fullTarget = np.concatenate((testTarget, trainTarget), axis = 0)
#
# fullParams = addColumnTetaZero(fullParams)
#
# normalParams = findNormalParams (fullParams, fullTarget)
#
# print (normalParams)
