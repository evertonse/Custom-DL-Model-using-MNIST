# load the mnist dataset
import numpy as np

def fetch(url):
  import requests, gzip, os, hashlib
  fp = os.path.join("/", hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    with open(fp, "wb") as f:
      dat = requests.get(url).content
      f.write(dat)
  return np.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()
  
def load_mnist_data():
	X = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
	Y = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
	X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28*28))
	Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

	training_data60k = {
		"inputs": np.array(X)/255.0,
		"outputs":np.array([np.array([y == int(label) for y in  range(10)]).astype(int) for label in Y])
	}

	testing_data10k = {
		"inputs": np.array(X_test)/255.0,
		"outputs":np.array([np.array([y == int(label) for y in  range(10)]).astype(int) for label in Y_test])
	}

	all_data = {
			"inputs": np.concatenate((training_data60k["inputs"],  testing_data10k["inputs"])),
			"outputs":np.concatenate((training_data60k["outputs"], testing_data10k["outputs"]))
		}
	return all_data
	