# Activities
* Pesquisar o melhor resultado do keggle e comentar dele

# Proposed solutions 
* Linear, correlação e graficos levando ao quadratico, remoção de alguns parametros?
* Botar gráficos obtidos

# Experiments

### Modelo puramente linear com todos os parâmetros
* Alphas, epochs, costs, tempo (para todas as formas)
  
### Modelo quadrático em x,y e z com todos os parâmetros
* Alphas, epochs, costs, tempo (para todas as formas)
  
### Modelo quadrático em x,y e z tirando depth e table
* Alphas, epochs, costs, tempo (para todas as formas)
  
###Alphas: 
* 0.01
* 0.001
* 0.0001
* 0.00001

###Epochs:
* 25000, 50000, 100000, 1000000
  
## Comparations
### SGDRegressor
* Epochs, alpha, cost, time

### Normal equation

#Conclusions
* Remoção de dados muito divergentes?
* Penalização de alguns thetas? Ver se os parâmetros adicionados prejudicaram.
* Aumentar o numero de iterações até ter determinada convergência (evitar iterações inuteis)


#References
Hands-On Machine Learning ([Chapter 2]((https://www.safaribooksonline.com/library/view/Hands-On+Machine+Learning+with+Scikit-Learn+and+TensorFlow/9781491962282/ch02.html#download_the_data) and  [Chapter 4]((https://www.safaribooksonline.com/library/view/Hands-On+Machine+Learning+with+Scikit-Learn+and+TensorFlow/9781491962282/ch02.html#download_the_data))) 
