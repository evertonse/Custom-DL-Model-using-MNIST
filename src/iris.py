
import sys
sys.path.append('./vendor')
sys.path.append('./src')
from vendor.csvdata import data
import numpy as np


def encode(lista: list):
	"5.1,3.5,1.4,0.2,Iris-setosa"
	petala_L, petala_W, sepala_L, sepala_W, specie = lista

	if specie == "Iris-setosa":
		output = (1,0,0)
	elif specie == "Iris-versicolor":
		output = (0,1,0)
	elif specie == "Iris-virginica":
		output = (0,0,1)

	else:
		print("[[UNKNOWN FLOWER SPECIE]]")
	return (
		(float(petala_L),
		float(petala_W),
		float(sepala_L),
		float(sepala_W)),
		output)

def load_iris_data():
	url = "./assets/iris.data"
	# Escolha de Linhas
	dados = data(url=url,filter=encode)
	
	training_data = {
		"inputs": np.array(dados["colunas"](0)[0]),
		"outputs":np.array(dados["colunas"](1)[1]),
		"dimensions": (4,8,8,8,3)
	}
	return training_data


"""Vamos ter 
	a Petala, 
	a Sepala("mini petala"), 
	Comprimento da Petala, 
	Largura da Petala, 
	Comprimentoda Sepala, 
	Largura de Sepala
	This is perhaps the best known database to be found in the pattern recognition literature.
	The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.
	One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.


5. Number of Instances: 150 (50 in each of three classes)

6. Number of Attributes: 4 numeric, predictive attributes and the class

7. Attribute Information:
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class: 
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica

8. Missing Attribute Values: None

Summary Statistics:
	         Min  Max   Mean    SD   Class Correlation
   sepal length: 4.3  7.9   5.84  0.83    0.7826   
    sepal width: 2.0  4.4   3.05  0.43   -0.4194
   petal length: 1.0  6.9   3.76  1.76    0.9490  (high!)
    petal width: 0.1  2.5   1.20  0.76    0.9565  (high!)

9. Class Distribution: 33.3% for each of 3 classes.

"""