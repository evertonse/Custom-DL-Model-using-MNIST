from src.TorchModel import TorchModel
import torchvision.datasets as datasets
import torch as tc
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision import transforms   
from torch import nn
	


# Se transformarmos para algo ao invez de 'None' o que acontece?
train_data 	= datasets.MNIST(root='./data', train=True, download=True, transform=None)
test_data 	= datasets.MNIST(root='./data', train=False, download=True, transform=None)

print(train_data[1][0])
print(train_data[0][1])
print(type(train_data[1]))
print(type(train_data[0]))


train_data = {
	"inputs"	: [np.array(x[0], np.float32, copy=False).reshape(28*28)/255.0 for x in train_data],
	"outputs": tc.tensor([[y[1] == x for x in range(10)] for y in train_data]).type(tc.int32)
}

test_data = {
	"inputs"	: [np.array(x[0], np.float32, copy=False).reshape(28*28)/255.0 for x in test_data],
	"outputs": tc.tensor([[y[1] == x for x in range(10)] for y in test_data]).type(tc.int32)
}

model = TorchModel(dimensions=(784,128,128,10), activation="relu")


epochs = 1  # Loss: 5.842959281500181
lr = 0.045
model.fit(
	train_data["inputs"],
	train_data["outputs"], 
	epochs=epochs, 
	lr=lr)


def test_prediction(index, data, model:nn.Module):
	current_image = data["inputs"][index]

	y_predict = model.predict(index, data["inputs"], data["outputs"])
	
	prediction = (y_predict == y_predict.max()).type(tc.int)

	guess = list(prediction).index(1)
	
	label = data["outputs"][index]
	ground_truth = list(label).index(1)

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


print()
path = "./model"
tc.save(model.state_dict(), path)
plt.plot(model.epoch_losses)
plt.plot(model.epoch_losses_train)

model = TorchModel(dimensions=(784,128,128,10), activation="relu")
model.load_state_dict(tc.load(path))
model.eval()

test_prediction(2, data=test_data,model=model)	

