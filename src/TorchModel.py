import torch as tc
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision import transforms   
from torch import nn

class TorchModel(nn.Module):
	def __init__(self,dimensions=(2,2,2,2), activation="sigmoid", output_activation="sigmoid"):
		super(TorchModel, self).__init__()

		# Change this to CUDA
		self.device 		= "cuda" if tc.cuda.is_available() else "cpu"
		self.loss 			= nn.MSELoss()
		self.optmizer 		= None
		self.losses 		= list()
		self.epoch_losses = list()
		self.accuracies 	= list()

		self = self.to(self.device)
		
		print(f"> Rodando tudo em {self.device}")
		
		#// TODO(Everton): Implement choice of activations
		if activation == "sigmoid":
			pass
		
		"""# // TODO(Everton): Implement change of dimensions
		self.layers = nn.Sequential(
			nn.Linear(784,128), 	# 1 layer:-> 784 input 128 o/p
			nn.ReLU(),          	# Defining Regular linear unit as activation
			nn.Linear(128,64),  	# 2 Layer:-> 128 Input and 64 O/p
			# Possivel estabelecer um droupout -> nn.Dropout(p = 0.4),
			nn.ReLU(),          	# Defining Regular linear unit as activation
			nn.Linear(64,10),   	# 3 Layer:-> 64 Input and 10 O/P as (0-9)
			nn.Sigmoid() # Defining the log softmax to find the probablities for the last output unit
      )"""
		layers = list()
		
		# Setting Interim Layers
		if activation.lower() == "relu":
			for idx in range(len(dimensions)-2):
				layers.append(nn.Linear(dimensions[idx],dimensions[idx + 1]))
				layers.append(nn.ReLU())

		elif activation.lower()  == "sigmoid":
			for idx in range(len(dimensions)-2):
				layers.append(nn.Linear(dimensions[idx],dimensions[idx + 1]))
				layers.append(nn.Sigmoid())

		elif activation.lower()  == "tanh":
			for idx in range(len(dimensions)-2):
				layers.append(nn.Linear(dimensions[idx],dimensions[idx + 1]))
				layers.append(nn.Tanh())
			
				

		# Setting output Layer
		layers.append(nn.Linear(dimensions[-2],dimensions[-1]))
		if output_activation.lower() == "sigmoid":
			layers.append(nn.Sigmoid())

		if output_activation.lower() == "softmax":
			layers.append(nn.Softmax())

		if output_activation.lower() == "relu":
			layers.append(nn.ReLU())

		# Unpack Layers
		self.layers = nn.Sequential(*layers)

	def forward(self, X):
		return self.layers(X)

	def fit(self, inputs, outputs,shuffle=True, autosave=False, epochs:int=10, lr=0.036):

		self.optmizer = tc.optim.SGD(self.parameters(), lr=lr)

		for e in range(epochs):
			if autosave:
				# Save each 20th epoch
				if e % 20 == 0:
					self.save(f"Model{e/20}")
					print(f"> epoch:{len(self.epoch_losses)+1}th, auto saved in : " + f"Model{e/20}")
			
			# A cada epoch calculo a media do erro de cada set de treinamento

			train_loss = self.__train(inputs,outputs, lr, shuffle=shuffle)
			self.epoch_losses.append(train_loss)
			self.losses = list()

			#self.losses.append(train_loss)

			if e % 1 == 0:
				print(f"Epoch: {len(self.epoch_losses)}th; Train Loss: {train_loss}; Acurracy: {self.accuracies[-1]}")

	
	def __train(self, inputs, outputs, lr, shuffle=True):	
		self.train()

		# __train corresponde a 1 epoch, sendo assim é o erro acumulado do epoch inteiro
		cumloss = 0.0
		itr_count = 0 
		
		# A cada epoch calculo a media do erro de cada set de treinamento
		predicts = list()
		
		if shuffle:
			assert len(inputs) == len(outputs)
			p = np.random.permutation(len(inputs))
			inputs,outputs = inputs[p], outputs[p]

		for index in range(len(inputs)):
			x = inputs[index]
			y = outputs[index]


			if tc.is_tensor(x):
				x = x.to(self.device)
			else:
				x = tc.tensor(x).to(self.device)
				
			x = x.type(tc.float32)
			
			if tc.is_tensor(y):
				y = y.to(self.device)
			else:
				y = tc.tensor(y).to(self.device)
			
			y = y.type(tc.float32)
			##type(X)
			# y = y.unsqueeze(784).float().to(device)		

			# Note that nn.Module objects are used as if they are functions 
			#(i.e they are callable), but behind the scenes 
			# Pytorch will call our forward method automatically.
			pred = self.__call__(x)
			loss = nn.MSELoss()(pred, y)

			#prediction = np.array([(guess == y_predict.max()).astype(int) for guess in y_predict]);
			prediction = (pred == pred.max()).type(tc.int)
			predicts.append(prediction)

			# loss = tc.criterion(X, y)		
			# zera os gradientes acumulados
			self.optmizer.zero_grad()
			# computa os gradientes
			loss .backward()
			# anda, de fato, na direção que reduz o erro local
			self.optmizer.step()		
			# loss é um tensor; item pra obter o float

			single_loss = loss.item()

			self.losses.append(single_loss)

			cumloss += single_loss
			itr_count += 1
		# retornamos o erro acumulado dividico pela quantidade de dados
		# mas antes adicionamos acurácia
		self.accuracies.append(self.accuracy(predicts, outputs)[0])
		return cumloss / len(inputs)
		
	def test(self, inputs, outputs):
		self.eval()
		print("> Entrou no Modo Teste")
		predicts = list()
		
		total 	= len(inputs)
		correct = 0 
		with tc.no_grad():
			
			for i in range(total):
				x = inputs[i]
				y = outputs[i]
				
				if tc.is_tensor(x):
					x = x.to(self.device)
				else:
					x = tc.tensor(x).to(self.device)
					
				x = x.type(tc.float32)
				
				if tc.is_tensor(y):
					y = y.to(self.device)
				else:
					y = tc.tensor(y).to(self.device)
				
				y = y.type(tc.float32)
				
				pred = self.__call__(x)


				prediction = (pred == pred.max()).type(tc.int)
				predicts.append(prediction)
				pred_index  = prediction.argmax()

				groud_truth_index = y.argmax()
				
				if (pred_index == groud_truth_index):
					correct += 1
		print(f"Accuracy of the network on the {total} test images: %d %%" % (100 * correct / total))	
		print(self.accuracy(predicts,outputs)[0])	
		
		return (100 * correct / total)

	def predict(self, x):
			
		self.eval()
		print("> Entrou no Modo Predicao")		
		with tc.no_grad():
			if tc.is_tensor(x):
				x = x.to(self.device)
			else:
				x = tc.tensor(x).to(self.device)	
			x = x.type(tc.float32)
			pred = self.__call__(x)
		return pred
		# they must be the same legnth

	# outputs must be hot encoded
	def accuracy(self,predictions,outputs):
		right_guesses = 0 
		total = len(predictions)
		wrong_indexes = list()

		for index, guess in enumerate(predictions):
			if np.array_equal(guess,outputs[index]):
				right_guesses += 1
			else:
				wrong_indexes.append(index)
		if(total > 0):
			return (right_guesses/total,wrong_indexes)
	
	def print(self):
		print(f"\nCurrent Loss: {self.epoch_losses[-1]}")
		print(f"\nCurrent Acurácia: {self.accuracies[-1]}\n")
		plt.title("Erro Médio por Epochs")
		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.plot(tc.arange(len(self.epoch_losses)), self.epoch_losses, color= "b", linestyle = "-", linewidth=1.25)		
		plt.plot(self.epoch_losses)
	