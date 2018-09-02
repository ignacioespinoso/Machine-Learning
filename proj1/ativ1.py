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
import matplotlib.pyplot as plt

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
            data[i][3] = 1
        elif (data[i][3] == 'SI2'):
            data[i][3] = 2
        elif (data[i][3] == 'SI1'):
            data[i][3] = 3
        elif (data[i][3] == 'VS2'):
            data[i][3] = 4
        elif (data[i][3] == 'VS1'):
            data[i][3] = 5
        elif (data[i][3] == 'VVS2'):
            data[i][3] = 6
        elif (data[i][3] == 'VVS1'):
            data[i][3] = 7
        elif (data[i][3] == 'IF'):
            data[i][3] = 8

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

def costFunction (params, thetas, target):
    m = params.shape[0]
    agaDeTheta = np.matmul (params, thetas)
    preSomatorio = agaDeTheta - target
    return (sumSquares(preSomatorio, m))

def sumSquares(array, m):
    square = array * array
    return (sum(square)/(2*m))
# Main function

with open('diamonds-dataset/diamonds-train.csv', 'rb') as f:
    reader = csv.reader(f)
    diamondsTrain = list(reader)
with open('diamonds-dataset/diamonds-test.csv', 'rb') as f:
    reader = csv.reader(f)
    diamondsTest = list(reader)


testParams, testTarget = formatArray(diamondsTest)
trainParams, trainTarget = formatArray(diamondsTrain)

# rows = np.rand/ms = trainParams[rows,:]
# print (rows)
# print(validationParams.shape[0])
# print(validationParams.shape[1])
# print(validationParams[:5])
trainParams = addColumnTetaZero(trainParams)
testParams = addColumnTetaZero(testParams)

# trainPartParams = trainParams[(trainParams.shape[0]/5):,:]
# trainValidParams = trainParams[:(trainParams.shape[0]/5),:]
#
# trainPartTarget = trainTarget[(len(trainTarget)/5):]
# trainValidTarget = trainTarget[:(len(trainTarget)/5)]
m = trainParams.shape[0]


thetas = createArrayTheta(10, 100)
alpha = 0.0002
# FAZER O FOR ATE DETERMINADO NUMERO DE ITERACOES OU ERRO < X

testTargetTranspose = np.asarray(testTarget)
testTargetTranspose = testTargetTranspose.reshape(testTargetTranspose.shape[0], -1)

trainTargetTranspose = np.asarray(trainTarget)
trainTargetTranspose = trainTargetTranspose.reshape(trainTargetTranspose.shape[0], -1)
xToPlot = np.array([])
yToPlot = np.array([])
i = 0
custo0 = 10
custo1 = 1


# while i < 25000 and abs(1 - custo1/custo0) > 0.9*alpha:
while i < 20000 :
    custo0 = custo1
    agaDeTheta = np.matmul (trainParams, thetas)
    preSomatorio = agaDeTheta - trainTargetTranspose
    somatorio = preSomatorio * trainParams
    somaTheta = np.sum(somatorio, axis=0)
    somaTheta = somaTheta.reshape(somaTheta.shape[0], -1)
    somaTheta = somaTheta / m
    somaTheta = somaTheta * alpha
    thetas = thetas - somaTheta
    i = i + 1
    custo1 = costFunction(trainParams, thetas, trainTargetTranspose)
    xToPlot = np.append(xToPlot,i)
    yToPlot = np.append(yToPlot,custo1)
    print (i, custo1/custo0, custo1)


results = np.matmul (trainParams, thetas)
print ()
print (results[:5])
print (trainTargetTranspose[:5])

plt.plot(xToPlot, yToPlot, 'ro')
plt.show()






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
# # ADICIONAR A COLUNA DE 1 e analisar os resultados
# trainParamsFit = fitParams (trainParams)
# testParamsFit = fitParams (testParams)
#
# fullParams = np.concatenate((testParams, trainParams), axis = 0)
# fullTarget = np.concatenate((testTarget, trainTarget), axis = 0)
#
# fullParams = addColumnTetaZero(fullParams)
# normalParams = findNormalParams (fullParams, fullTarget)
#
# print (normalParams)
# print (costFunction(fullParams, normalParams, fullTarget))
