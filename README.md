# CS6910 Assignment 1
## Author: Akshay Charjan CS22M008

###Report Link: https://wandb.ai/cs22m008/Asg%201%20Final/reports/CS6910-Assignment-1--VmlldzozODI3MTUx?accessToken=l0f5y6q0dc5721qo13nok3hp162voc7xjp0featazzsogeiffm6fxmpg7elyq00r

## Instructions to create a model, train, predict Neural Network
Install the required libraries.

### Instructions to create a model, use the below funtion: 
```
obj = model(args.num_layers,args.hidden_size,args.optimizer, args.activation,args.weight_init,args.weight_decay, args.epsilon)
```
### To train the model, use the below function:
```
obj.train(trainX,trainY,valX, valY, args.batch_size,args.epochs,args.momentum,args.beta,args.beta1,args.beta2,args.learning_rate,1,args.loss)
```
The parameters the model takes in: 
 ```
numLayers: The number of hidden layers in the neural network (excluding the input layer)
numNeurons: The number of neurons in each hidden layer of the neural network
optimizer: The optimization algorithm used to update the weights and biases during training (e.g. SGD, Adam, etc.)
activation_funtion: The activation function used in each neuron of the neural network (e.g. sigmoid, ReLU, etc.)
initialization: The method used to initialize the weights and biases of the neural network (e.g. random, Xavier, etc.)
l2_lambda: The regularization parameter used to prevent overfitting in the neural network
epsilon: A small value used to prevent numerical instability in the optimization algorithm
```
### To add a new activation function:

Create  a function for the activation function and its derivative as the class model and call it in the class as mentioned by a comment.
 
### To predict, use the predict function to get the predicted Y values. 
obj.predict(testX)

### To get the accuracy, get_accuracy funtion is provided: 
obj.get_accuracy(testY,y_pred_test.T)
