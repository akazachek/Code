import os
import urllib
import pickle
import torch
import gzip
from torchvision import datasets
from torchvision import transforms
from neural_net import dirichlet_net

#########
# MNIST-M data loading adapted from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cogan/mnistm.py
#########

MNIST_M_URL = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"
MNIST_M_NAME = "keras_mnistm.pkl.gz"
MNIST_M_PATH = "./data/mnistm/"
MNIST_M_FILE = MNIST_M_PATH + MNIST_M_NAME
MNIST_PATH = "./data/mnist/"

#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"

# download MNIST-M if not already present
os.makedirs(MNIST_M_PATH, exist_ok=True)
os.makedirs(MNIST_PATH, exist_ok=True)
data = urllib.request.urlopen(MNIST_M_URL)
if not os.path.exists(MNIST_M_FILE):
    with open(MNIST_M_FILE, "wb") as f:
        f.write(data.read())

# open and read MNIST-M data
with gzip.open(MNIST_M_FILE, "rb") as f:
    mnist_m = pickle.load(f, encoding="bytes")
mnist_m_train_x = torch.ByteTensor(mnist_m[b"train"]) / 255.
mnist_m_test_x = torch.ByteTensor(mnist_m[b"test"]) / 255.

# flatten rows and columns of images
num_train = mnist_m_train_x.shape[0]
mnist_m_train_x = mnist_m_train_x.reshape((num_train, 28*28*3)).to(DEVICE)
num_test = mnist_m_test_x.shape[0]
mnist_m_test_x = mnist_m_test_x.reshape((num_test, 28*28*3)).to(DEVICE)

# load regular MNIST dataset
mnist_train = datasets.MNIST(root=MNIST_PATH, train=True, download=True)
mnist_test = datasets.MNIST(root=MNIST_PATH, train=False, download=True)

# convert to tensors
num_train = len(mnist_train)
mnist_train_x = torch.empty((num_train, 28, 28))
mnist_train_y = torch.empty((num_train)).to(DEVICE)
for i in range(num_train):
    img, label = mnist_train[i]
    mnist_train_x[i] = transforms.ToTensor()(img)
    mnist_train_y[i] = label

num_test = len(mnist_test)
mnist_test_x = torch.empty((num_test, 28, 28))
mnist_test_y = torch.empty((num_test)).to(DEVICE)
for i in range(num_test):
    img, label = mnist_test[i]
    mnist_test_x[i] = transforms.ToTensor()(img)
    mnist_test_y[i] = label

# flatten rows and columns of images
mnist_train_x = mnist_train_x.reshape((num_train, 28*28))
mnist_test_x = mnist_test_x.reshape((num_test, 28*28))

# MNIST data is only greyscale, so to get dimensions to
# match MNIST-M we duplicate each greyscale value 3 times
mnist_train_x = mnist_train_x.repeat_interleave(3, dim=1).to(DEVICE)
mnist_test_x = mnist_test_x.repeat_interleave(3, dim=1).to(DEVICE)

layers = [28*28*3, 50, 50, 10]
model = dirichlet_net(layers, mnist_train_x, mnist_train_y,
                      mnist_m_train_x).to(DEVICE)

model.train(num_iter=1000, batch_size=16,
            eta=1e7, log=True, dirichlet=True)

model.test(mnist_test_x, mnist_test_y, log=True)
model.test(mnist_m_test_x, mnist_test_y, log=True)
