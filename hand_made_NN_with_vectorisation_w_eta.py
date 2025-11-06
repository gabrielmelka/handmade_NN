# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 12:23:22 2025

@author: gabriel melka
"""

#%% importations of libraries

import numpy as np
import struct
from os.path import join
import matplotlib.pyplot as plt
import time

#%%

# Sigmoïde appliquée élément par élément
def sigma(Z):
    return 1 / (1 + np.exp(-Z))

# Dérivée de la sigmoïde
def deriv_sigma(Z):
    S = sigma(Z)
    return S * (1 - S)

# Softmax appliqué colonne par colonne (chaque colonne = une image)
def softmax(Z):
    eZ = np.exp(Z)                 # exponentielle de chaque élément
    return eZ / np.sum(eZ, axis=0, keepdims=True)  # normalisation par colonne

# Fonction de coût cross-entropy pour un batch
def L_cost(pred, true):
    # pred, true : shape (n_out, m)
    return -np.sum(true * np.log(pred + 1e-30)) / pred.shape[1]  # moyenne par image

    
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


r = 1224  # for example
image, label = show_image_at_index(x_train, y_train, r)
print(f"The image of index {r} corresponds to the digit {label}")

#%% parameters



n_samples = 5000  # number of random images
training_length=5000 #number of iterations in the training
list_eta=[0.5, 0.8, 1.1]
list_L_eta=[]
list_accuracy_scores_eta=[]
list_time_scoring_eta=[]
time_total=[]


#%% initialise matrix and resizing data, and training now

for eta in list_eta:
    List_L=[]
    
    N0=28*28 #size of the original MNIST images
    N1=36
    N2=16
    N3=10
    
    accuracy_scores=[]
    
    W1 = np.random.randn(N1, N0) / np.sqrt(N0) #the normalisation of Xavier to accelerate the convergence
    W2 = np.random.randn(N2, N1) / np.sqrt(N1)
    W3 = np.random.randn(N3, N2) / np.sqrt(N2)
    
    
    B1 = np.random.randn(N1) 
    B2 = np.random.randn(N2)
    B3 = np.random.randn(N3)
    
    total_images = training_length * n_samples 
    
     # we transform the square image into a single column 
    x_train_flat = x_train.reshape(x_train.shape[0], -1)  # shape (60000, 676)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    len_test=len(x_test_flat)
    
    start_total = time.time()
    start_iter = time.time()
    time_scoring=[]
    
    if len(accuracy_scores)!=0:
        raise ValueError("Error : execute the previous cell please !!")
    
    for k in range(training_length):
        # the scoring
        if (k%50==0 and k<500) or k <10 or (k<100 and k%5==0) or k%200==0 or k==training_length-1 :
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
                for j in range(10):
                    if pred[j]>pred[ind]:
                        ind=j
                if y_test[i]==ind:
                    accuracy+=1
            time_scoring.append(k)
            accuracy_scores.append(accuracy)
        
        # 500 random images
        indices = np.random.choice(60000, size=n_samples, replace=False)
        
        
        X_batch = x_train_flat[indices].T          # (N0, n_samples)
        Y_batch = y_train[indices]                 # (n_samples,)
    
        # one-hot encoding vectorisé
        Y_true = np.zeros((10, n_samples))
        Y_true[Y_batch, np.arange(n_samples)] = 1  # chaque colonne = un label
        
    
        # forward pass to calculate L
        X1 = W1 @ X_batch + B1[:, None]     # (N1, n_samples)
        Z1 = sigma(X1)
        
        X2 = W2 @ Z1 + B2[:, None]          # (N2, n_samples)
        Z2 = sigma(X2)
        
        X3 = W3 @ Z2 + B3[:, None]          # (N3, n_samples)
        Z3 = softmax(X3)                    # (N3, n_samples)
        
        List_L.append(np.mean(L_cost(Z3, Y_true)))
    
        #  backward pass
    
        delta3 = Z3 - Y_true                            # (N3, n_samples)
        delta2 = (W3.T @ delta3) * deriv_sigma(X2)      # (N2, n_samples)
        delta1 = (W2.T @ delta2) * deriv_sigma(X1)      # (N1, n_samples)
    
        
        dW3 = (delta3 @ Z2.T) / n_samples               # (N3, N2)
        dW2 = (delta2 @ Z1.T) / n_samples               # (N2, N1)
        dW1 = (delta1 @ X_batch.T) / n_samples          # (N1, N0)
        
        dB3 = np.mean(delta3, axis=1)                   # (N3,)
        dB2 = np.mean(delta2, axis=1)                   # (N2,)
        dB1 = np.mean(delta1, axis=1)                   # (N1,)
    
        # the gradient descent
        W1 -= eta * dW1
        W2 -= eta * dW2
        W3 -= eta * dW3
        B1 -= eta * dB1
        B2 -= eta * dB2
        B3 -= eta * dB3
        
        elapsed_iter = time.time() - start_iter
        
        
        print(f"Iteration {k+1}/{training_length} terminée en {elapsed_iter:.2f}s — eta={eta}")
    
    elapsed_total = time.time() - start_total
    print(f"Temps total pour {training_length} itérations : {elapsed_total:.2f}s")
    time_total.append(elapsed_total)
    list_L_eta.append(List_L)
    list_accuracy_scores_eta.append(accuracy_scores)
    list_time_scoring_eta.append(time_scoring)
    


#%% final results on scoring data
colors = ['#7f7f7f','tab:blue', 'tab:purple' , 'tab:brown', 'tab:red' ,'#87CEEB', 'tab:orange','tab:green', '#000000', '#90EE90']
# Premier graphique
plt.figure(figsize=(8,5))

for i in range(len(list_eta)): 
    plt.plot(list_L_eta[i], 'o', markersize=4, label=f'eta={list_eta[i]}', color=colors[i])
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.text(0.7, 0.7, f"nn_samples = {n_samples}",
             fontsize=10,
             verticalalignment='center', 
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
             transform=plt.gca().transAxes,  
             clip_on=False)
plt.title("Loss during training")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid(True, linestyle=':', linewidth=0.5)
plt.legend()
plt.show()


# Deuxième graphique
plt.figure(figsize=(8,5))
for i in range(len(list_eta)): 
    plt.plot(np.array(list_time_scoring_eta[i]), np.array(list_accuracy_scores_eta[i])/len_test*100, 'x',color=colors[i], markersize=5, label=f'eta={list_eta[i]}')
plt.axhline(10, color='red', linestyle='--', linewidth=1, label='10% limit (random)')
plt.axhline(100, color='orange', linestyle='--', linewidth=1, label='100%')
plt.title("Accuracy during training")
plt.xlabel("Iterations")
plt.ylabel("Accuracy (%)")
plt.text(0.7, 0.7, f"eta = {eta}\nn_samples = {n_samples}",
         fontsize=10,
         verticalalignment='center',
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
         transform=plt.gca().transAxes,  
         clip_on=False)  
plt.legend()
plt.show()


#to see if the cost-function is decreasing as we want
#%%
## follow-up : look at the final accuracy depending on the eta factor !!

accuracy=accuracy/len_test*100
print("accuracy :" , accuracy, "%") 