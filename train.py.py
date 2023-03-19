import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
# pip install wandb
from keras.datasets import fashion_mnist, mnist
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('-wp','--wandb_project',default ='myprojectname',metavar="",required = False,type=str,help = "Project name used to track experiments in Weights & Biases dashboard" )
parser.add_argument('-we','--wandb_entity',default ='myname',metavar="",required = False,type=str,help = "Wandb Entity used to track experiments in the Weights & Biases dashboard." )
parser.add_argument('-d','--dataset',default='fashion_mnist',metavar="",required = False,type=str,help ="" ,choices= ["mnist", "fashion_mnist"])
parser.add_argument('-e','--epochs',default=1,metavar="",required = False,type=int,help = "Number of epochs to train neural network.")
parser.add_argument('-b','--batch_size',default=4,metavar="",required = False,type=int,help = "Batch size used to train neural network.")
parser.add_argument('-l','--loss',default='cross_entropy',metavar="",required = False,type=str,help ="",choices= ["mean_squared_error", "cross_entropy"])
parser.add_argument('-o','--optimizer',default='sgd',metavar="",required = False,type=str,help = "",choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"])
parser.add_argument('-lr','--learning_rate',default=0.1,metavar="",required = False,type=float,help = "Learning rate used to optimize model parameters")
parser.add_argument('-m','--momentum',default=0.5,metavar="",required = False,type=float,help = "Momentum used by momentum and nag optimizers.")
parser.add_argument('-beta','--beta',default=0.5,metavar="",required = False,type=float,help = "Beta used by rmsprop optimizer")
parser.add_argument('-beta1','--beta1',default=0.5,metavar="",required = False,type=float,help = "Beta1 used by adam and nadam optimizers.")
parser.add_argument('-beta2','--beta2',default=0.5,metavar="",required = False,type=float,help = "Beta2 used by adam and nadam optimizers.")
parser.add_argument('-eps','--epsilon',default=0.000001,metavar="",required = False,type=float,help = "Epsilon used by optimizers.")
parser.add_argument('-w_d','--weight_decay',default=.0,metavar="",required = False,type=float,help = "Weight decay used by optimizers.")
parser.add_argument('-w_i','--weight_init',default='random',metavar="",required = False,help ="", choices= ["random", "Xavier"])
parser.add_argument('-nhl','--num_layers',default=1,metavar="",required = False,type=int,help ="Number of hidden layers used in feedforward neural network.")
parser.add_argument('-sz','--hidden_size',default=4,metavar="",required = False,type=int, help = "Number of hidden neurons in a feedforward layer.")
parser.add_argument('-a','--activation',default='sigmoid',metavar="",required = False, help = "Activation Function", choices= ["identity", "sigmoid", "tanh", "ReLU"])
args=parser.parse_args()

if args.dataset=="fashion_mnist":
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
else:
    (trainX, trainY), (testX, testY) = mnist.load_data()

class_labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

wandb.login(key="a2e6402ce9fe2ebe1f01d5332c4fafa210b0dc0c")
pName = "Final Run ultimate 7"
run_obj=wandb.init( project=pName)

trainX = trainX.reshape(trainX.shape[0], 784)
testX = testX.reshape(testX.shape[0], 784)
#feature Scaling
trainX=trainX/255.0
testX=testX/255.0
# Split the X_train into a training set and validation set
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.1, random_state=100)

def softmax(a):
    return np.exp(a)/np.sum(np.exp(a),axis=0)

def grad_softmax(x):
    return softmax(x) * (1-softmax(x))

def sigmoid(a):
    return 1.0 / (np.exp(-a)+1.0)

def grad_sigmoid(a):
  return sigmoid(a)*(1-sigmoid(a))

def tanh(x):
    return np.tanh(x)

def grad_tanh(x):
    return (1 - (np.tanh(x)**2))

def relu(x):
    return np.maximum(0,x)

def grad_relu(x):
    return 1*(x>0) 

def identity(x):
  return x

def grad_identity(x):
  return np.ones(x.shape)

class model():
  def __init__(self,numLayers,numNeurons,optimizer,activation_funtion,initialization,l2_lambda, epsilon): 
    self.numLayers=numLayers+1 
    self.numHiddenLayers=self.numLayers-1
    self.numNeurons=numNeurons
    self.numClasses=10
    self.grad_w=[]
    self.grad_b=[]
    self.y_pred=[]
    self.u_w=0
    self.u_b=0
    self.m_w=0
    self.m_b=0
    self.W_L=[]
    self.b_L=[]
    self.optimizer=optimizer
    self.epsilon=epsilon
    if(activation_funtion=="sigmoid"):
      self.g=sigmoid
      self.grad_activation=grad_sigmoid
    if(activation_funtion=="ReLU"):
      self.g=relu
      self.grad_activation=grad_relu
    if(activation_funtion=="tanh"):
      self.g=tanh
      self.grad_activation=grad_tanh
    if(activation_funtion=="identity"):
      self.g=identity
      self.grad_activation=grad_identity     

    self.initialization=initialization
    self.l2_lambda=l2_lambda
    self.W=[]
    self.b=[]
      

  def initialize(self):
    #Initialising weights and Biases
    if self.initialization=="random":
      W = []
      W.append((np.random.uniform(-1,1,(784,self.numNeurons))))
      for i in range (2 , self.numHiddenLayers+1):
        W.append((np.random.uniform(-1,1,(self.numNeurons,self.numNeurons))))
      W.append((np.random.uniform(-1,1,(self.numNeurons,self.numClasses))))
      self.W= np.array(W)

    if self.initialization=="Xavier":
      W = []
      W.append((np.random.uniform(-1,1,(784,self.numNeurons)))*np.sqrt(2/(784+self.numNeurons)))
      for i in range (2 , self.numHiddenLayers+1): 
        W.append((np.random.uniform(-1,1,(self.numNeurons,self.numNeurons)))*np.sqrt(2/(self.numNeurons+self.numNeurons)))
      W.append((np.random.uniform(-1,1,(self.numNeurons,self.numClasses)))*np.sqrt(2/(self.numNeurons+self.numClasses)))
      self.W= np.array(W)

    b = []
    for i in range (1 , self.numLayers): 
      b.append(np.zeros((self.numNeurons,1)))
    b.append(np.zeros((self.numClasses,1)))
    self.b= np.array(b)

  def back_propagation(self,Y,batch_size,loss):
    grad_a=[None]*(self.numLayers)
    grad_b=[None]*(self.numLayers)
    grad_h=[None]*(self.numLayers)
    grad_w=[None]*(self.numLayers)
    oneHot_y=self.compute_oneHot_y(Y)

    h=self.activation
    a=self.preActivation
    W=self.W

    if loss=="cross_entropy":
      grad_a[self.numLayers-1]=self.y_pred-oneHot_y.T 
    elif loss=="mean_squared_error":
      grad_a[self.numLayers-1]=(self.y_pred-oneHot_y.T)*grad_softmax(a[self.numLayers-1])

   
    for k in range (self.numLayers-1,-1,-1): 
      grad_w[k]=np.matmul(grad_a[k], h[k].T)+self.l2_lambda*W[k].T
      grad_b[k]=np.sum(grad_a[k],axis=1,keepdims=True)/batch_size++self.l2_lambda*self.b[k]
      grad_h[k]=np.matmul(W[k],grad_a[k])

      if(k>0):
        grad_a[k-1] =grad_h[k] * self.grad_activation(a[k-1])
    self.grad_b,self.grad_w=grad_b,grad_w

  def feed_forward(self,X):
    a=[None]*(self.numLayers)
    h=[None]*(self.numLayers)
    k=0
    h[0]=X.T
    for k in range(0, self.numLayers-1): 
      a_k=self.b[k]+np.matmul(self.W[k].T,h[k]) 
      h_k=self.g(a_k)
      a[k]=(a_k)
      h[k+1]=(h_k)
    a[self.numLayers-1]=self.b[self.numLayers-1]+np.matmul(self.W[self.numLayers-1].T,h[self.numLayers-1])

    output=softmax(a[self.numLayers-1])

    self.activation,self.preActivation=h,a
    self.y_pred=output
    return output

  def train(self,trainX,trainY,valX, valY, batch_size,epochs,momentum,beta,beta_1,beta_2,neta,t,loss):
    self.initialize()

    if self.optimizer=='nag':
      self.init_nag()
    epoch_train_loss=[]
    epoch_val_loss=[]

    for j in range(0,epochs):
      for i in range(0, trainX.shape[0],batch_size):
       
        if (self.optimizer == 'nag'):
          W_copy=self.W
          b_copy=self.b
          self.pre_update_nag(momentum)

        self.feed_forward(trainX[i:i+batch_size])
        self.back_propagation(trainY[i:i+batch_size],batch_size,loss)
        Grad_w=np.array(self.grad_w)
        for i in range(0,Grad_w.shape[0]):
          Grad_w[i]=Grad_w[i].T
        Grad_b=np.array(self.grad_b)

        if (self.optimizer == 'sgd'):
          self.update_sgd(neta,Grad_w,Grad_b)
        if (self.optimizer == 'momentum'): 
          self.update_mom(neta,momentum,Grad_w,Grad_b)
        if (self.optimizer == 'nag'):
          self.update_nag(neta,momentum,Grad_w,Grad_b,W_copy,b_copy)
        if (self.optimizer == 'rmsprop'):
          self.update_rmsprop(neta,beta,Grad_w,Grad_b)
        if (self.optimizer == 'adam'):
          self.update_adam(neta,beta_1,beta_2,Grad_w,Grad_b,t)
        if (self.optimizer == 'nadam'):
          self.update_nadam(neta,beta_1,beta_2,Grad_w,Grad_b,t)

        #self.update_new_optimiser()     The new optimiser algorithm will be called here

        t+=1

      self.feed_forward(trainX)
      train_acc=self.get_accuracy(trainY,self.y_pred.T)*100
      print("train accuracy: ",train_acc)
      train_loss=self.compute_loss(self.y_pred,self.compute_oneHot_y(trainY),trainX.shape[0],loss)
      val_loss, val_accuracy=self.test( valX, valY, beta, neta,loss)


      print("Epoch number: ", j+1, "\tTraining loss:", train_loss)
      epoch_train_loss.append(train_loss)
      epoch_val_loss.append(val_loss)
      wandb.log({"train_accuracy ": train_acc, "Training loss ": train_loss, "val_accuracy": val_accuracy, "val_loss":val_loss, "epochs ": j+1})

    plt.plot(list(range(len(epoch_train_loss))), epoch_train_loss, 'r', label="Training loss")
    plt.plot(list(range(len(epoch_val_loss))), epoch_val_loss, 'b', label="val loss")

    plt.title("Training Loss and Valdation loss vs Number of Epochs", size=14)
    plt.xlabel("Number of epochs", size=14)
    plt.ylabel("Loss", size=14)
    plt.grid()
    plt.legend()
    plt.show()


  def test(self,valX,valY,beta,neta,loss):
    self.feed_forward(valX)

    val_loss=self.compute_loss(self.y_pred,self.compute_oneHot_y(valY),valX.shape[0],loss)

    val_accuracy=self.get_accuracy(valY,self.y_pred.T)*100
    print("val accuracy: ", val_accuracy, "\t val loss:", val_loss)
    return val_loss, val_accuracy

  def predict(self,TestX):
    output = self.feed_forward(TestX)
    return output


  def update_sgd(self,neta,Grad_w,Grad_b):
        self.W=self.W-neta*Grad_w
        self.b=self.b-neta*Grad_b
  
  def update_mom(self,neta,momentum,Grad_w,Grad_b):
        self.u_w=momentum*self.u_w+(1-momentum)*Grad_w
        self.u_b=momentum*self.u_b+(1-momentum)*Grad_b
        self.W=self.W-neta*self.u_w
        self.b=self.b-neta*self.u_b
  
  def update_nag(self,neta,momentum,Grad_w,Grad_b,W_copy,b_copy):
        self.W=W_copy
        self.b=b_copy
        self.u_w=momentum*self.u_w+(1-momentum)*Grad_w
        self.u_b=momentum*self.u_b+(1-momentum)*Grad_b
        self.W=self.W-neta*self.u_w
        self.b=self.b-neta*self.u_b


  def init_nag(self):
    W_L = []
    W_L.append((np.zeros([784,self.numNeurons])))
    for i in range (2 , self.numHiddenLayers+1): #Hiddenlayer 1 to last hidden layer (starts from 2 coz first layer is init just above)
      W_L.append((np.zeros([self.numNeurons,self.numNeurons])))
    W_L.append((np.zeros([self.numNeurons,self.numClasses])))
    W_L= np.array(W_L)

    b_L = []
    for i in range (1 , self.numLayers): #Hiddenlayer1 to last hidden layer
      b_L.append(np.zeros((self.numNeurons,1)))
    b_L.append(np.zeros((self.numClasses,1)))
    b_L= np.array(b_L)
    self.W_L=W_L
    self.b_L=b_L

  def pre_update_nag(self,momentum):
    self.W_L=self.W-momentum*self.u_w
    self.b_L=self.b-momentum*self.u_b
    self.W=self.W_L
    self.b=self.b_L

  def update_rmsprop(self,neta,beta,Grad_w,Grad_b):
        self.u_w=beta*self.u_w+(1-beta)*Grad_w*Grad_w
        self.u_b=beta*self.u_b+(1-beta)*Grad_b*Grad_b
        self.W=self.W-(neta/(self.u_w**0.5+self.epsilon))*Grad_w
        self.b=self.b-(neta/(self.u_b**0.5+self.epsilon))*Grad_b
  
  def update_adam(self,neta,beta_1,beta_2,Grad_w,Grad_b,t):

        self.m_w=beta_1*self.m_w+(1-beta_1)*Grad_w
        self.m_b=beta_1*self.m_b+(1-beta_1)*Grad_b

        self.u_w=beta_2*self.u_w+(1-beta_2)*Grad_w*Grad_w
        self.u_b=beta_2*self.u_b+(1-beta_2)*Grad_b*Grad_b

        m_w_hat=self.m_w/(1-beta_1**t)
        m_b_hat=self.m_b/(1-beta_1**t)

        self.W=self.W-(neta/(self.u_w**0.5+self.epsilon))*m_w_hat
        self.b=self.b-(neta/(self.u_b**0.5+self.epsilon))*m_b_hat

  def update_nadam(self,neta,beta_1,beta_2,Grad_w,Grad_b,t):

        self.m_w=beta_1*self.m_w+(1-beta_1)*Grad_w
        self.m_b=beta_1*self.m_b+(1-beta_1)*Grad_b

        self.u_w=beta_2*self.u_w+(1-beta_2)*Grad_w*Grad_w
        self.u_b=beta_2*self.u_b+(1-beta_2)*Grad_b*Grad_b

        m_w_hat=self.m_w/(1-beta_1**t)
        m_b_hat=self.m_b/(1-beta_1**t)

        self.W=self.W-(neta/(self.u_w**0.5+self.epsilon))*(beta_1*m_w_hat+((1-beta_1)*Grad_w)/(1-beta_1**t))
        self.b=self.b-(neta/(self.u_b**0.5+self.epsilon))*(beta_1*m_b_hat+((1-beta_1)*Grad_b)/(1-beta_1**t))

  """"def update_new_optimiser(): 
        Update function for a new optimisation algorithm to be entered here """

  def compute_oneHot_y(self,Y):
    oneHot_y=[]
    for i in range(0,Y.shape[0]):
      temp=np.zeros(self.numClasses)
      temp[Y[i]]=1
      oneHot_y.append(temp)
    oneHot_y=np.array(oneHot_y)
    return oneHot_y

  def compute_loss(self,y_pred,oneHot_y,numImages,loss):

    loss_reg=0
    for i in range(0, self.W.shape[0]):
        loss_reg += np.sum(np.square(np.linalg.norm(self.W[i])))
        loss_reg += np.sum(np.square(np.linalg.norm(self.b[i])))
    if loss == 'cross_entropy':
        return (-1.0 * np.sum(np.multiply(oneHot_y.T, np.log(y_pred)))/numImages + (self.l2_lambda*loss_reg/2))

    elif loss == 'mean_squared_error':
        return  ((1/2) * np.sum((oneHot_y.T-y_pred)**2)/numImages + (self.l2_lambda*loss_reg/2))


  def get_accuracy(self,Y_true,Y_pred):
    size=Y_true.shape[0]
    corrects = 0
    s=0
    while(s<size):
      t=0
      maxT = 0
      maxS = 0

      while (t<10):
        if (maxS < Y_pred[s][t]):
          maxT = t
          maxS = Y_pred[s][t]
        t+=1
      if (maxT == Y_true[s]):
        corrects+=1
      accuracy=corrects/size
      s+=1
    return accuracy


obj = model(args.num_layers,args.hidden_size,args.optimizer,args.activation,args.weight_init,args.weight_decay, args.epsilon)
obj.train(trainX,trainY,valX, valY, args.batch_size,args.epochs,args.momentum,args.beta,args.beta1,args.beta2,args.learning_rate,1,args.loss)

wandb.finish()
