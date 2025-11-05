# handmade_NN
this small personal project is about building entirerly by myself a Neural Network (NN), and optimising it via a gradient descent. The goal of the project is to classify MNIST images of digits with the best accuracy (and speed) possible.

the repositary contains two documents, the vectorisation one being the successor of the "wo" without vectorisation
the description of the script is simple, we take iamges (28 x 28 size) from the MNIST database ( 50000 of training and 10000 of testing), and we build by hand a NN, with 2 hidden layers of 36 and 16 neurons. Then we optimise the weights and biases of the neurons using a gradient descent on them.

the loss function is a Shannon-type loss-function that punishes heavily the wrong answers 

activations functions are sigmoid (that's really arbitrarly from me), and the last activation function is obviously a softmax, so that we can consider that the last vector represents the probability of each digit representing the image


The two versions :


- the first version is gross : there is a loop on each image of a random batch of 500 images, and we calculate the contribution of each image on the dB and dB, the derivative of the loss function with respect to the weights and biases. then we take the mean of this derivative (matrix), and we proceed to change the weights and the biases thanks to a gradient descent. It's super slow ( around 2 minutes for 50 itération of 500 batches of training). We could say that this is the CPU approach (even though we are already using matrix at each step)
- to correct that we are using vectorisation: after having translated each vector into a vector (one column), we are putting together those column to get a big matrix (size 784=28*28 x size of the batch=500). Then we proceed all the calculations on this thing. The formulas are a bit more complex, but really close to the previous ones 


Results : 
- 1st script : around 60% accuracy for a training of 50 iterations with a batch of 500
- 2nd script : way faster, with around 2minutes in my PC, I can get training batches of 5000, and 2000 iterations. there is a increase of 600 times the speed I think. the results are also way better, the accuracy is generaly between 93-95% which good for a neural network that do not cares about the geometry of the image (non convolutionnal)


What I would like to get better
- I would like to verify that when the NN is wrong (it's not detcting the right digit), if the second best probabilty is usually right (ie : significantly better than in 1 out of 9 case), and also if the first probability (linked with the probability guess) is significantly lower than the average on all the batch (it would be horrible that there is lots of cases where the NN guessed at a 99.99% chance that it's a 2 but in fact it's an 8)
- I would like to train for differents eta (hyperparameter for the speed of the gradient descent), to see which is the best accuracy-wise (if there is clearly one), and if that is repercuting one the loss function (so that we can change the parameter during the training). I would also want to know if the optimum eta_best is somewhere linked by a power rule to the number of iterations in the training, because my intition tells me that when N_training is big, eta can be low because the training has a lot of time to find the global minimum of the loss-function
- I then would like to do some CNN to get an even better accuracy.
