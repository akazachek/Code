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

mnist_train_x_2 = mnist_test_x[:10000, ]
mnist_train_y_2 = mnist_test_y[:10000]
mnist_train_x = mnist_test_x[10000:, ]
mnist_train_y = mnist_test_y[10000:]

layers = [28*28*3, 50, 50, 10]
nu = 1e1
eta = 1.
first_iters = 3501
second_iters = 3501
batch = 16

# model training without hypothesis risk (just cross-entropy)
# at start, then refined using hypothesis-risk
print("==================== model 1 ====================")
print("starting first phase...\n")
model = dirichlet_net(layers, mnist_train_x, mnist_train_y,
                      mnist_m_train_x).to(DEVICE)
model.train(num_iter=first_iters, batch_size=batch, nu=nu,
            eta=eta, log=True, dirichlet=False)

print("\ntesting on source...")
model.test(mnist_test_x, mnist_test_y, log=True)
print("\ntesting on target...")
model.test(mnist_m_test_x, mnist_test_y, log=True)

print("\nstarting second phase...\n")
model.train(num_iter=second_iters, batch_size=batch, nu=nu,
            eta=eta, log=True, dirichlet=True)
print("\ntesting on source...")
model.test(mnist_test_x, mnist_test_y, log=True)
print("\ntesting on target...")
model.test(mnist_m_test_x, mnist_test_y, log=True)

# model training only using cross-entropy
print("==================== model 2 ====================")
print("starting first phase...\n")
model2 = dirichlet_net(layers, mnist_train_x, mnist_train_y,
                       mnist_m_train_x).to(DEVICE)
model2.train(num_iter=first_iters, batch_size=batch, nu=nu,
             eta=eta, log=True, dirichlet=False)

print("\ntesting on source...")
model2.test(mnist_test_x, mnist_test_y, log=True)
print("\ntesting on target...")
model2.test(mnist_m_test_x, mnist_test_y, log=True)

print("\nstarting second phase...\n")
model2.train(num_iter=second_iters, batch_size=batch, nu=nu,
             eta=eta, log=True, dirichlet=False)
print("\ntesting on source...")
model2.test(mnist_test_x, mnist_test_y, log=True)
print("\ntesting on target...")
model2.test(mnist_m_test_x, mnist_test_y, log=True)

# model training using cross-entropy and dirichlet cost
# the whole time
print("==================== model 3 ====================")
print("starting first phase...\n")
model3 = dirichlet_net(layers, mnist_train_x, mnist_train_y,
                       mnist_m_train_x).to(DEVICE)
model3.train(num_iter=first_iters, batch_size=batch, nu=nu,
             eta=eta, log=True, dirichlet=True)

print("\ntesting on source...")
model3.test(mnist_test_x, mnist_test_y, log=True)
print("\ntesting on target...")
model3.test(mnist_m_test_x, mnist_test_y, log=True)

print("\nstarting second phase...\n")
model3.train(num_iter=second_iters, batch_size=batch, nu=nu,
             eta=eta, log=True, dirichlet=True)

print("\ntesting on source...")
model3.test(mnist_test_x, mnist_test_y, log=True)
print("\ntesting on target...")
model3.test(mnist_m_test_x, mnist_test_y, log=True)
