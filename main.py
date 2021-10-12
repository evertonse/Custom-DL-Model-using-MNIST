import numpy as np
import sys
sys.path.append('./src')

from digits import digits_data
from vendor.Model import Model 
from iris import iris_data
from digits import digits_data
import matplotlib.pyplot as plt

def __main__():
	traning_data 	= dict()
	testing_data	= dict()
	dimensions 		= tuple()

	#model = "xor"
	model = "iris"
	model = "digits"
	

	if model == "iris":
		traning_data = iris_data()
		dimensions = traning_data["dimensions"]
	if model == "digits":
		traning_data,testing_data, all_data, dimensions = digits_data()
		print("all_data input length", len(all_data["inputs"]))
		print("all_data outputs length", len(all_data["outputs"]))

	menu(training_data=traning_data,testing_data=testing_data, dims=dimensions)
	

def test_prediction(index, data, model:Model):
	current_image = data["inputs"][index]
	y_predict = model.predict(current_image)[0]
	prediction = (y_predict == y_predict.max()).astype(int)

	guess = list(prediction).index(1)

	label = data["outputs"][index]
	print("Label: ", label)
	print("Prediction: ", prediction)

	plt.gray()
	plt.title("Modelo acha que é: " + str(guess))
	plt.imshow( current_image.reshape((28, 28)) * 255, interpolation='nearest')
	#plt.show()
	plt.draw()
	plt.waitforbuttonpress(0) # this will wait for indefinite time		
	plt.close()


def menu(training_data:dict,testing_data, dims = (2,8,8,2)):

	#load_filename = "Model v1.0" # Oficial Mais bem treinado
	load_filename = ""
	save_filename = ""


	learning_rate = 0.035
	epochs = 10

	print(training_data["inputs"][1].__len__())
	print(testing_data["inputs"][1].__len__())
	print(training_data["inputs"].__len__())
	print(testing_data["inputs"].__len__())
	
	model  = Model((784,128,128,10), activation="sigmoid",verbose=0, wr=(-0.5,0.5) )
	

	while True:
		get = input("\n> Digite um comando: (ex: 'train','quit','print','guess','lr' ,'epoch', 'save' ,'load','test')\n")

		if get == "":
			print("\n> Model Started Training...\n")
			
			model.train(
				training_data["inputs"],
				training_data["outputs"],
				lr = learning_rate, epochs=epochs,
				shuffle=True,
				autosave=True)
			
			print("> Done.")
			model.print()

		elif get == "lr":
			try:
				learning_rate = float(input("> Digite novo Learning Rate: "))	
			except:
				print("Seu input é insatisfatório, digite um número")
				continue
		elif get[0].lower() == 'e':
			try:
				epochs = int(input("> Digite nova quantidade de épocas: "))
			except:
				print("Seu input é insatisfatório, digite um número")
				continue

		# quit
		elif get[0].lower() == 'q':
			break;
		
		# print
		elif get[0].lower() == 'p':
			model.print()
			continue;
		
		# guess
		elif get[0].lower()  == 'g':
			predicts = list()
			for y in testing_data["inputs"]:
				y_predict = model.predict(y)[0]
				prediction = (y_predict == y_predict.max()).astype(int)
				predicts.append(prediction)
			accuracy,wrong_indexes = model.accuracy(predictions=predicts, outputs=testing_data["outputs"],verbose=1)
			print("Indexes onde está errado: ",wrong_indexes)
			print("Acurácia: ",accuracy )
			continue;
		
		# save
		elif get.lower()  == "save":
			if save_filename == "":
				save_filename = input("> save as:")

			model.save("./models/" + save_filename)
			print("> model saved in: ",save_filename)
			continue;

		elif get.lower() == "load":
			if load_filename == "":
				load_filename = input("> load as:")			

			model = Model.load("./models/" + load_filename)
			print("> model loaded from: ",load_filename)
			continue

		elif get.lower() == "test":
			while True:
				index = input("> Escolha uma imagem entre [0, 10k): ")
				if not index.isnumeric():
					break
				try:
					test_prediction(int(index),testing_data, model)
				except:
					print("> Imagem deve ser entre 1 e 10'000\n")
				continue
			continue

		else:
			continue

			
__main__();

