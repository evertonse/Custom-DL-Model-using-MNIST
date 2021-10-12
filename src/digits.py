
import sys
sys.path.append('./vendor')
sys.path.append('./src')
from vendor.csvdata import data
import numpy as np
import pickle

# Global , caso seja primeira vez rodando deve ser setado para 'True', 
# irá gerar binários pelo pickle e vc após gerar vc deve setar para False
first_time_running = False

def encode(lista: list):
	"label, ...pixel"
	label, *pixels = lista
	one_hot_label = np.array([y == int(label) for y in  range(10)]).astype(int)

	return (np.array(pixels).astype(float), one_hot_label)

def save_mnist_data(filename:str,content_data) :
	with open("./assets/digit-recognizer/" + filename + ".pickle", "wb") as file:
		pickle.dump(content_data, file)

def load_mnist_data(filename:str) :
	with open("./assets/digit-recognizer/" + filename + ".pickle", "rb") as file:
		loaded_data = pickle.load(file)
		return loaded_data


def digits_data():
	url_reduced		= "./assets/digit-recognizer/train_reduced.csv"
	url_train40k	= "./assets/digit-recognizer/train.csv"
	url_train60k	= "./assets/mnist/mnist_train.csv"
	url_test10k		= "./assets/mnist/mnist_test.csv"


	# Escolha de Linhas
	if first_time_running:
		raw_40k	= data(url=url_train40k,filter=encode)
		raw_60k	= data(url=url_train60k,filter=encode)
		raw_10k		= data(url=url_test10k,filter=encode)

		all_data = dict()
		training_data40k = {
			"inputs": np.array(raw_40k["colunas"](0)[0])/255.0,
			"outputs":np.array(raw_40k["colunas"](1)[1]),
		}

		training_data60k = {
			"inputs": np.array(raw_60k["colunas"](0)[0])/255.0,
			"outputs":np.array(raw_60k["colunas"](1)[1]),
		}

		testing_data10k = {
			"inputs": np.array(raw_10k["colunas"](0)[0])/255.0,
			"outputs":np.array(raw_10k["colunas"](1)[1]),
		}
		all_data = {
			"inputs":np.concatenate( (training_data60k["inputs"],  testing_data10k["inputs"])),
			"outputs":np.concatenate((training_data60k["outputs"], testing_data10k["outputs"]))
		}
		print("all_data input length", len(all_data["inputs"]))

		save_mnist_data("train_40k",training_data40k)
		save_mnist_data("train_60k",training_data60k)
		save_mnist_data("test_10k",testing_data10k)
		save_mnist_data("all_data",all_data)

		


	# LOADING DESERIALIZATION
	else:
		training_data40k 	=	load_mnist_data("train_40k")
		training_data60k	=	load_mnist_data("train_60k")
		testing_data10k	=	load_mnist_data( "test_10k")
		all_data				=	load_mnist_data( "all_data")
	
	
	return (training_data40k, testing_data10k, all_data, (784,128,128,10))
