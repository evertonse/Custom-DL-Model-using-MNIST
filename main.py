from src.mnist import load_mnist_data
from src.Model import Model
import matplotlib.pyplot as plt


def test_prediction(index, data, model:Model):
	current_image = data["inputs"][index]
	y_predict = model.predict(current_image)[0]
	prediction = (y_predict == y_predict.max()).astype(int)

	guess = list(prediction).index(1)
	label = data["outputs"][index]
	ground_truth = list(label).index(1)

	print("Label: ", label)
	print("Prediction: ", prediction)

	# Opção de desobrigar de fornecer label correto, para quando formor utilizar paint
	if len(label) < 10:
		label = "made on paint"
		ground_truth = " paint"

	print("Label: ", label)
	print("Prediction: ", prediction)
	plt.gray()
	plt.title("Model thinks it is: " + str(guess) + "\nGround truth: " + str(ground_truth))
	plt.imshow( current_image.reshape((28, 28)) * 255, interpolation='nearest')
	plt.xticks([])
	plt.yticks([0])
	plt.show()

	
def __main__():


	all_data = load_mnist_data()

	print("Quantidade de exemplos:",	len(all_data["inputs"]))
	print("Dimensão da imagem: ",		len(all_data["inputs"][0]))
	print("Quantidade de digitos: ",	len(all_data["outputs"][0]))

	# Treinamos com 42 mil exemplos
	train_data = {
		"inputs" :	all_data["inputs" ][:42000],
		"outputs":	all_data["outputs"][:42000]
	}

	# Testamos com restante 28 mil exemplos
	test_data = {
		"inputs" :	all_data["inputs" ][42000:],
		"outputs":	all_data["outputs"][42000:]
	}

	learning_rate = 0.035
	# 3 épocas é bem pouco, mas já chega a 95% de acurácia, podendo chega a 98.9% com mais treinamento
	epochs = 3
	model_filename = "model_128x128"

	model  = Model((784,128,128,10), activation="sigmoid", verbose=0, wr=(-0.5,0.5))
	#model = Model.load("./models/" + model_filename)

	print("\n> Model Started Training...\n")
				
	model.train(
		train_data["inputs"],
		train_data["outputs"],
		lr = learning_rate, epochs=epochs,
		shuffle=True,
		autosave=False)

	print("> Done.")

	model.print()

	model.save("./models/" + model_filename)
	print("> model saved in: ",model_filename)


	while True:
		index = input("> Escolha uma imagem entre [0, 28'000): ")
		if not index.isnumeric():
			break
		try:
			test_prediction(int(index),test_data, model)
		except:
			print("> Imagem deve ser entre 1 e 28'000\n")
		continue

__main__()