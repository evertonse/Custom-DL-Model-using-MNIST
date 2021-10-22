from PIL.Image import NONE
import numpy as np					# Math
import matplotlib.pyplot as plt	# Gráficos
import pickle							# Serialização 



#/*===========================HELPER FUNCTIONS START=======================*/



# [Sigmoid funciona melhor com os dados normalizados, entre [0,1] ]
def sigmoid(x:int, derivative = False):
	if derivative == True:
		# o x da derivada é baseado no y da função original
		return x * (1.0 - x)
	#return np.e**x /( (np.e**x)	+  1)
	return 1 /(1 + np.exp(-x))
	

# Ativação varia entre (-1, 1)
# Mais testes ainda precisam serem executados
def tanh(x:int, derivative = False):
	if derivative == True:
		# o x da derivada é baseado no y da função original
		# SE NÃO FOR USADO ASSIM DA ERRADO
		return (1.0 - (x**2))
	return np.tanh(x)

# [[NÃO USE, não testada ainda]]
def relu(x:int, derivative = False):
	if derivative == True:
		# o x da derivada é baseado no y da função original
		return (x > 0).astype(float)
	return np.maximum(x, 0)


	
# Cost Function: (y¹ - y¹')^2 + (y² - y²')^2 
# this considers the cost for one single pass a one vector from the training_data
def loss( guess, expected):
	custo = 0
	for index, value in enumerate(expected):
		custo = custo + (guess[index] - value )**2
	return custo

# Trocar ordem dos dados mantendo ordem entre input e output (X,Y)
def unison_shuffled_copies(a, b):
	assert len(a) == len(b)
	p = np.random.permutation(len(a))
	return a[p], b[p]



#/*===========================HELPER FUNCTIONS END=======================*/

# Esse Modelo permite quantidades arbitrárias de Hiddens Layer, especificados por dimensions
# Activation function é sigmoid
# output é uma probabilidade feita pela função softmax <- MENTIRA NÃO FIZ ISSO
class Model:
	# Escolhmemos a dimensão de cada layer
	def __init__(self,dimensions=(2,2,2,2) , activation="sigmoid", verbose=0, wr=(-1,1)):#wr = wheights range
		# dimensions = (input_dim:int, hidden1_dim:int, hidden2_dim:int, output_dim:int)
		input_dim	= dimensions[0]
		output_dim	= dimensions[-1] 
		#information things
		if activation == "sigmoid":
			print("> activation: 'sigmoid' selected")

			self.activate = sigmoid
			self.y_activate = sigmoid
			
		elif activation == "tanh":
			print("> activation: 'tanh' selected")
			self.activate = tanh
			self.y_activate = tanh

		elif activation == "relu":
			print("> activation: 'relu' selected")
			self.activate = relu
			self.y_activate = sigmoid

		else:
			print("> activation: 'NONE' selected")

		self.losses 		= list()
		self.epoch_losses = list()
		self.accuracies 	= list()
		self.verbose = verbose

		# X np.array of  inputs
		self.X	=	 np.ones(input_dim).reshape(1, input_dim)
		# Y np.array of  expected outputs for each input
		self.Y	=	 np.ones(output_dim).reshape(1, output_dim)
				
		self.Hs	=	 [np.ones(dim).reshape(1, dim) for dim in dimensions[1:-1]]
		
		# All of these start beatween -1 to 1
		# Wheights and bias from the input to  hidden
		self.Ws = list()
		self.Bs = list()
		for i in range(len(dimensions)-1):
			self.Ws.append(np.random.uniform(wr[0],wr[1], size=(dimensions[i], dimensions[i+1])))

		for i in range(len(dimensions)-1):
			self.Bs.append(np.random.uniform(wr[0],wr[1], size=(1, dimensions[i+1])))

	# Passamos os dados pela network e setamos um Y = output

	def __feed_foward(self, inputs: list):
		self.X		= np.array(inputs).reshape(self.X.shape)

		self.Hs[0]	= self.activate(np.dot(self.X,self.Ws[0]) +  self.Bs[0]) #

		for i in np.arange(len(self.Hs)-1) + 1:
			self.Hs[i]	= self.activate(np.dot(self.Hs[i-1],self.Ws[i]) + self.Bs[i]) #

		#self.H2	= self.activate(np.dot(self.H1,self.W2) + self.B2) #

		self.Y = self.y_activate(np.dot(self.Hs[-1],self.Ws[-1]) + self.Bs[-1]) #

		

	def __adjust_wheights(self, expected: list, lr:float = 0.001):
		Y_expected = np.array(expected).reshape(self.Y.shape)
		
		
		# output erro do esperado em comparação com o que a NN advinhou

		# contribuição do erro da hidden layer
		# levando em conta a contribuição de cada peso da hidden -> output
		Y_error = Y_expected - self.Y

		Y_delta = 2*Y_error*self.y_activate(self.Y, derivative=True)

		Hs_erros	 = list(range(len(self.Hs)))
		Hs_deltas = list(range(len(self.Hs)))
		
		# Primeiro erro é calculado de maneira diferente
		Hs_erros[-1] = Y_delta @ self.Ws[-1].T

		for i in np.arange(len(self.Hs)-1)[::-1]:
			Hs_erros[i] = Hs_erros[(i + 1)] @ self.Ws[(i + 1)].T

		for i in np.arange(len(self.Hs))[::-1]:
			Hs_deltas[i] = 2*Hs_erros[i]*self.activate(self.Hs[i], derivative=True)

		# Losses é irrelevante para o treinamento, é apenas para analise
		loss = np.sum((Y_error**2))/(self.Y.shape[0]*self.Y.shape[1])
		self.losses.append(loss)


		# A derivada da função custo com relação aos pesos hidden -> output
		# multiplicado pela transporta da hiden layer para gerar um delta_peso_W2 da mesma dimensão do peso_W2
		# first Bias and wheigts base on the Y
		#// TODO(Everton): Make Y just be another Layer and X also
		self.Ws[-1] += lr*(self.Hs[-1].T @ Y_delta )
		self.Bs[-1] += lr*Y_delta

		for i in np.arange(len(self.Ws))[1:-1][::-1]:
			self.Ws[i] +=  lr*(self.Hs[i-1].T @ Hs_deltas[i])
		
		for i in np.arange(len(self.Bs))[:-1][::-1]:
			self.Bs[i] +=  lr*Hs_deltas[i]

		self.Ws[0]  += lr*(self.X.T  @ Hs_deltas[0])

	# Aqui é o treinamento de apenas 1 input
	def __train_once(self, input, output, lr):
		self.__feed_foward(input)
		self.__adjust_wheights(output,lr)

	def train(self, inputs , outputs, lr:int, epochs:int,shuffle=True, autosave=False):
		if epochs < 1:
			return
		for i in range(epochs):
			# Damos um shuffle nos dados mantendo output e input pareado
			X,Y = inputs, outputs
			if shuffle:
				X,Y = unison_shuffled_copies(np.array(inputs), np.array(outputs))

			#rand = np.random.randint(0,len(inputs)-batchsize)
			#assert rand > 0

			for index in range(len(inputs)):
				x = X[index].reshape(self.X.shape)
				y = Y[index].reshape(self.Y.shape)
				
				if self.verbose > 1:
					print(f"x: {x} y: {y}")
				
				self.__train_once(x, y, lr)
				
			# Save per iteration of the whole training data
			if autosave:
				# Save each 20th epoch
				if i % 20 == 0:
					self.save(f"Model{i/20}")
					print(f">epoch:{i}th, auto saved in : " + f"Model{i/20}")
		
			# A cada epoch calculo a media do erro de cada set de treinamento
			self.epoch_losses.append(np.sum(self.losses)/len(self.losses))
			predicts = list()
			for y in inputs:
				y_predict = self.predict(y)[0]
				#prediction = np.array([(guess == y_predict.max()).astype(int) for guess in y_predict]);
				prediction = (y_predict == y_predict.max()).astype(int)
				predicts.append(prediction)

			self.accuracies.append(self.accuracy(predicts, outputs)[0])
		self.accuracy(predicts, outputs,verbose=self.verbose)
			
	# they must be the same legnth
	# outputs must be hot encoded
	def accuracy(self,predictions,outputs,verbose=0):
		right_guesses = 0 
		wrong_indexes = list()
		if verbose==1:
				print(">   Guess / Output :")
		for index, guess in enumerate(predictions):
			#print("My guesses and outputs: ",guess,outputs[index])
			if verbose==1:
				print("> ",guess, outputs[index])
			if np.array_equal(guess,outputs[index]):
				right_guesses += 1
			else:
				wrong_indexes.append(index)
		return (right_guesses/len(predictions),wrong_indexes)

	# Tenta adivinha um apenas
	# Muito overlap entre outras funções, ainda precisa de melhoras
	def predict(self, input,verbose=0):
		input = np.array(input).reshape(self.X.shape)

		if verbose != 0:
			print(f"INPUT escolhido:\n{input}\n")
		
		self.__feed_foward(input)

		if verbose != 0:
			print("Prediction:")
			print([[f"{y:.8f}" for y in y_lista]  for y_lista in self.Y ])
			
		return self.Y

	
	# Maneira de visualizar pesos, bias e camadas
	def print(self, verbose=0):
		print("/*==================================================*/")
		if verbose > 0:
			for i in range(len(self.Ws)):	
				print(f"W{i+1} (wheights {i+1})\n{self.Ws[i]}")
				print(f"B{i+1} (bias {i+1}) :\n{self.Bs[i]}\n")

		print(f"X  (input) layer:\n{self.X}")
		for i in range(len(self.Hs)):	
			print(f"H{i+1} (hidden {i+1}) LAYER:\n{self.Hs[i]}")
		
		print(f"Y  (output) LAYER:\n{self.Y}\n")
		print(f"\nCurrent Loss: {self.epoch_losses[-1]}")
		print(f"\nCurrent Acurácia: {self.accuracies[-1]}\n")
	
		print("/*==================================================*/");

		plt.xlabel("Epochs")
		plt.ylabel("Loss")
		plt.plot(np.arange(len(self.epoch_losses)), self.epoch_losses, color= "b", linestyle = "-", linewidth=1.25)		
		plt.show()

	# Utiliza Pickle para Salvar
	def save(self,filename,compact=False):
		temp_list= list()
		with open(filename + ".pickle", "wb") as file:
			if compact == True:
				temp_list = [self.epoch_losses,self.losses,self.accuracies]
				self.epoch_losses = self.epoch_losses
				self.losses = [self.losses[-1]]
				self.accuracies = [self.accuracies[-1]]
			pickle.dump(self, file)
		if compact == True:
			self.epoch_losses, self.losses,self.accuracies = temp_list
		print("> model saved in: ",filename)
	
	# Utiliza Pickle dar Load
	def load(filename):
		with open(filename + ".pickle", "rb") as file:
			return pickle.load(file)

