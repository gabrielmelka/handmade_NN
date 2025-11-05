# -*- coding: utf-8 -*-
"""
Éditeur de Spyder*

@author: gabriel melka

"""
#%% importations of libraries

import numpy as np
import struct
from os.path import join
import matplotlib.pyplot as plt
import time

#%% fonctions that I will need

def sigma(L):
    n=len(L)
    s=np.zeros(n)
    for i in range(n):
        s[i]=1/(1+np.exp(-L[i]))
    return s

def softmax(L):
    n=len(L)
    norm=0
    for i in range(n):
        norm+=np.exp(L[i])
    s=np.zeros(n)
    for i in range(n):
        s[i]=np.exp(L[i])/norm
    return s
    
def deriv_sigma(L):
    return np.array([sigma(L)[i]*(1-sigma(L)[i]) for i in range(len(L))])

def L_cost(pred, true): #the cost function, of parameters the prediction (an array of probabilities),
#and of the true (an array full of 0 with one 1 )
    return -np.sum(true * np.log(pred))

    
#%% just some information about my architechture (in english)

# so I take the MNIST databse of hand-written single digits, and I try to classify them, using a 
# neural network
# to reduce the complexity of my algorythm, I transfer them into 12x12 images, which is a vector 
# of size 144 (its not convolutionnal for now)
# I will use three hidden layers of size 36 an 16, and 10


#%% data importation and batching : training/scoring data
# everything here comes from the "READ MNIST DATA" on Kaggle, 

class MnistDataloader:
    def __init__(self, train_img, train_lbl, test_img, test_lbl):
        self.train_img = train_img
        self.train_lbl = train_lbl
        self.test_img = test_img
        self.test_lbl = test_lbl

    def read_images_labels(self, images_path, labels_path):
        with open(labels_path, 'rb') as f:
            _, size = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        with open(images_path, 'rb') as f:
            _, size, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols)
        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.train_img, self.train_lbl)
        x_test, y_test = self.read_images_labels(self.test_img, self.test_lbl)
        return (x_train, y_train), (x_test, y_test)


def show_image_at_index(x_data, y_data, r):
    plt.imshow(x_data[r], cmap='gray')
    plt.title(f'Index {r} — Label {y_data[r]}')
    plt.axis('off')
    plt.show()
    return x_data[r], y_data[r]

def sample_subset(x_data, y_data, n):
    idx = np.random.choice(len(x_data), n, replace=False)
    return x_data[idx], y_data[idx]

# data imports
input_path = r"C:\Users\melka\Downloads\MNIST_dataset"
train_img = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
train_lbl = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_img = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_lbl = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist = MnistDataloader(train_img, train_lbl, test_img, test_lbl)
(x_train, y_train), (x_test, y_test) = mnist.load_data()


r = 1234  # for example
image, label = show_image_at_index(x_train, y_train, r)
print(f"The image of index {r} corresponds to the digit {label}")


#%% generation of parameters 
N0=28*28 #size of the original MNIST images
N1=36
N2=16
N3=10


W1 = np.random.randn(N1, N0)
W2 = np.random.randn(N2, N1)
W3 = np.random.randn(N3, N2)


B1 = np.random.randn(N1)
B2 = np.random.randn(N2)
B3 = np.random.randn(N3)

#%% optimisation : gradient descent
eta = 0.5
List_L=[]
n_samples = 500  # number of random images 

training_length=50

start_total = time.time()
total_images = training_length * n_samples 

 # we transform the square image into a single column 
x_train_flat = x_train.reshape(x_train.shape[0], -1)  # shape (60000, 676)

for k in range(training_length):
    
    indices = np.random.choice(60000, size=n_samples, replace=False)
    
    start_iter = time.time()
    
    List_L_partial=[]
    
    sum_dB1 = np.zeros(N1)
    sum_dB2 = np.zeros(N2)
    sum_dB3 = np.zeros(N3)

    sum_dW1 = np.zeros((N1, N0))
    sum_dW2 = np.zeros((N2, N1))
    sum_dW3 = np.zeros((N3, N2))
    
    counter = 0  # to count 
    
    for r in indices :
        counter += 1
        
        image = x_train_flat[r] 
        label = y_train[r]
        
        #we transform the label into the one-hot encoder
        
        true=[0 for i in range(10)]
        for i in range(10):
            if i==label:
                true[i]=1
        true = np.array(true)
        
        #forward pass to calculate L :
            
        Z0=image
        
        X1 = W1 @ Z0 + B1
        X1=X1.flatten()
        Z1 = sigma(X1)
        
        X2 = W2 @ Z1 +B2
        X2=X2.flatten()
        Z2 = sigma(X2)
        
        X3 = W3 @ Z2 +B3
        X3=X3.flatten()
        Z3 = softmax(X3).T
        
        pred = Z3
        
        
        List_L_partial.append(L_cost(pred, true))
        
        #back propagation 
        
        delta3 = Z3-true #true is the one-hot label, the layer close to the end
        delta2 = W3.T @ delta3 * deriv_sigma(X2) # @ normal matrix product, * hadamard product
        delta1 = W2.T @ delta2 * deriv_sigma(X1)
        
        sum_dW3 += delta3.reshape(-1,1) @ Z2.reshape(1,-1) # dérivative of L respective to W3
        sum_dW2 += delta2.reshape(-1,1) @ Z1.reshape(1,-1)
        sum_dW1 += delta1.reshape(-1,1) @ Z0.reshape(1,-1)
    
        
        sum_dB1 += delta1 # dérivative of L respective to B1
        sum_dB2 += delta2
        sum_dB3 += delta3
        
        if counter % 20 == 0:
            images_done = k * n_samples + counter
            percent_done = 100 * images_done / total_images
            elapsed_total = time.time() - start_total
            print(f"Iteration {k+1}/{training_length}, image {counter}/{n_samples}, "
                  f"temps écoulé : {elapsed_total:.2f}s, progression {percent_done:.1f}%")
        
    List_L.append(sum(List_L_partial)/n_samples)  
    
    # the gradient descent, for every image
        
    W1 -= eta * sum_dW1/n_samples
    W2 -= eta * sum_dW2/n_samples
    W3 -= eta * sum_dW3/n_samples
        
    B1 -= eta * sum_dB1/n_samples
    B2 -= eta * sum_dB2/n_samples
    B3 -= eta * sum_dB3/n_samples
    
    elapsed_iter = time.time() - start_iter
    print(f"Iteration {k+1} terminée en {elapsed_iter:.2f}s")
    
elapsed_total = time.time() - start_total
print(f"Temps total pour {training_length} itérations : {elapsed_total:.2f}s")

#%% final results on scoring data 

plt.figure()
plt.plot(List_L)
plt.show()

#to see if the cost-function is decreasing as we want 
#%%
## follow-up : look at the accuracy on the test batch 
start_total = time.time()
x_test_flat = x_test.reshape(x_test.shape[0], -1)

len_test=len(x_test_flat)
accuracy=0
for i in range(len_test):
    ind=0
    image=x_test_flat[i]
    Z0=image
    
    X1 = W1 @ Z0 + B1
    X1=X1.flatten()
    Z1 = sigma(X1)
    
    X2 = W2 @ Z1 +B2
    X2=X2.flatten()
    Z2 = sigma(X2)
    
    X3 = W3 @ Z2 +B3
    X3=X3.flatten()
    Z3 = softmax(X3).T
    
    pred = Z3
    for k in range(10):
        if pred[k]>pred[ind]:
            ind=k
    if y_test[i]==ind:
        accuracy+=1

accuracy=accuracy/len_test*100
print("accuracy :" , accuracy, "%")
    
    
    
    
    
    